# -*- coding: utf-8 -*-
"""
Простой графический интерфейс для генерации кандидат-разметки очагов РС.

Врач выбирает папку с исследованиями (DICOM или FLAIR .nii/.nii.gz), отмечает
галочками нужные исследования и нажимает «Запустить». Модель находит очаги
(nnU-Net + дополнение 2.5D-детектором) и раскладывает результат для проверки
в 3D Slicer. Никаких команд и консоли.
"""
import os, sys, queue, threading, subprocess, tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "detect_for_review.py"
PYEXE = sys.executable
DEFAULT_OUT = ROOT / "output"
sys.path.insert(0, str(ROOT))

NOISE = ("it/s", "it]", "%|", "\r")


def is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    return any(tok in line for tok in NOISE) or s.startswith(("0%", "100%"))


class CheckList(ttk.Frame):
    """Прокручиваемый список исследований с галочками."""
    def __init__(self, master, rows=8):
        super().__init__(master)
        self.canvas = tk.Canvas(self, height=rows * 22, highlightthickness=1,
                                highlightbackground="#ccc")
        self.sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.sb.pack(side="right", fill="y")
        self.vars = {}
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 120), "units")

    def set_items(self, items):
        for w in self.inner.winfo_children():
            w.destroy()
        self.vars.clear()
        for name, kind in items:
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.inner, variable=v, text=f"{name}    [{kind}]").pack(anchor="w")
            self.vars[name] = v

    def selected(self):
        return [n for n, v in self.vars.items() if v.get()]

    def set_all(self, val):
        for v in self.vars.values():
            v.set(val)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Разметка очагов РС — для 3D Slicer")
        root.geometry("820x720")
        root.minsize(720, 620)
        self.q: "queue.Queue" = queue.Queue()
        self._build()
        self.root.after(100, self._drain)

    def _build(self):
        pad = dict(padx=10, pady=5)
        top = ttk.Frame(self.root); top.pack(fill="x", **pad)
        ttk.Label(top, text="Обнаружение очагов РС и предразметка I/P/S/U для 3D Slicer",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(top, foreground="#555",
                  text="1) Выберите папку. 2) Отметьте исследования. 3) «Запустить». "
                       "4) Откройте результат в 3D Slicer и проверьте разметку.").pack(anchor="w")

        frm = ttk.Frame(self.root); frm.pack(fill="x", **pad)
        ttk.Label(frm, text="Папка с исследованиями:").grid(row=0, column=0, sticky="w")
        self.in_var = tk.StringVar()
        self.in_entry = ttk.Entry(frm, textvariable=self.in_var, width=64)
        self.in_entry.grid(row=0, column=1, sticky="we", padx=6)
        # авто-сканирование, если путь вписали вручную (Enter или уход из поля)
        self.in_entry.bind("<Return>", lambda e: self._scan_current())
        self.in_entry.bind("<FocusOut>", lambda e: self._scan_current())
        ttk.Button(frm, text="Обзор…", command=self._pick_in).grid(row=0, column=2)
        ttk.Button(frm, text="Показать", command=self._scan_current).grid(row=0, column=3, padx=(4,0))
        ttk.Label(frm, text="Папка для результатов:").grid(row=1, column=0, sticky="w", pady=4)
        self.out_var = tk.StringVar(value=str(DEFAULT_OUT))
        ttk.Entry(frm, textvariable=self.out_var, width=64).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Обзор…", command=self._pick_out).grid(row=1, column=2)
        ttk.Label(frm, text="Мин. размер очага, мм³:").grid(row=2, column=0, sticky="w")
        self.mm_var = tk.StringVar(value="15")
        ttk.Entry(frm, textvariable=self.mm_var, width=8).grid(row=2, column=1, sticky="w", padx=6)
        frm.columnconfigure(1, weight=1)

        # список исследований
        head = ttk.Frame(self.root); head.pack(fill="x", **pad)
        ttk.Label(head, text="Исследования (галочка = обрабатывать):",
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(head, text="Снять все", command=lambda: self.checks.set_all(False)).pack(side="right")
        ttk.Button(head, text="Выбрать все", command=lambda: self.checks.set_all(True)).pack(side="right", padx=6)
        self.scan_lbl = ttk.Label(head, foreground="#777", text="выберите папку…")
        self.scan_lbl.pack(side="left", padx=10)
        self.checks = CheckList(self.root, rows=8)
        self.checks.pack(fill="both", expand=False, **pad)

        # опции
        opt = ttk.Frame(self.root); opt.pack(fill="x", **pad)
        self.det_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, variable=self.det_var,
                        text="Дополнять находки 2.5D-детектором (меньше пропусков)"
                        ).pack(anchor="w")
        self.cls_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, variable=self.cls_var,
                        text="Авто-класс кандидатов (I/P/S/U предположительно; иначе — метка «?»)"
                        ).pack(anchor="w")

        # кнопки
        btns = ttk.Frame(self.root); btns.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btns, text="▶  Запустить", command=self._run)
        self.run_btn.pack(side="left")
        self.open_btn = ttk.Button(btns, text="Открыть результаты", command=self._open_out, state="disabled")
        self.open_btn.pack(side="left", padx=8)
        self.pb = ttk.Progressbar(btns, mode="indeterminate", length=200)
        self.pb.pack(side="right")
        self.status = ttk.Label(self.root, text="Готов к запуску.", foreground="#333")
        self.status.pack(anchor="w", padx=12)

        logf = ttk.Frame(self.root); logf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(logf, height=8, wrap="word", font=("Consolas", 9))
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ------- выбор папок / сканирование -------
    def _pick_in(self):
        d = filedialog.askdirectory(title="Папка с исследованиями (DICOM или NIfTI)")
        if d:
            self.in_var.set(d)
            self._scan(d)

    def _scan_current(self):
        """Сканировать путь, вписанный вручную (Enter / уход из поля / «Показать»)."""
        raw = self.in_var.get().strip().strip('"')
        if not raw:
            return
        p = Path(raw)
        if not p.is_dir():
            alt = ROOT.parent / raw            # относительный путь от корня проекта
            if alt.is_dir():
                p = alt; self.in_var.set(str(alt))
        if not p.is_dir():
            self.scan_lbl.config(text="папка не найдена")
            return
        if getattr(self, "_last_scanned", None) == str(p):
            return
        self._last_scanned = str(p)
        self._scan(str(p))

    def _pick_out(self):
        d = filedialog.askdirectory(title="Папка для результатов")
        if d:
            self.out_var.set(d)

    def _scan(self, d):
        self.scan_lbl.config(text="сканирую…")
        self.checks.set_items([])
        threading.Thread(target=self._scan_worker, args=(d,), daemon=True).start()

    def _scan_worker(self, d):
        try:
            from detect_for_review import scan_studies
            items = scan_studies(Path(d))
            self.q.put(("__STUDIES__", items))
        except Exception as e:
            self.q.put(("__STUDIES__", []))
            self.q.put(f"[!] Не удалось прочитать папку: {e}")

    # ------- запуск -------
    def _open_out(self):
        try:
            os.startfile(self.out_var.get())
        except Exception:
            messagebox.showinfo("Папка результатов", self.out_var.get())

    def _log(self, text):
        self.log.insert("end", text + "\n"); self.log.see("end")

    def _run(self):
        indir = self.in_var.get().strip()
        if not indir or not Path(indir).is_dir():
            messagebox.showwarning("Проверьте ввод", "Выберите существующую папку с исследованиями.")
            return
        names = self.checks.selected()
        if not names:
            messagebox.showwarning("Проверьте ввод", "Отметьте хотя бы одно исследование.")
            return
        try:
            mm = float(self.mm_var.get().replace(",", "."))
        except ValueError:
            messagebox.showwarning("Проверьте ввод", "Мин. размер должен быть числом (мм³).")
            return
        out = self.out_var.get().strip() or str(DEFAULT_OUT)
        Path(out).mkdir(parents=True, exist_ok=True)

        sel_file = Path(tempfile.gettempdir()) / "ms_review_select.txt"
        sel_file.write_text("\n".join(names), encoding="utf-8")

        self.run_btn.config(state="disabled"); self.open_btn.config(state="disabled")
        self.pb.start(12)
        self.status.config(text=f"Обработка {len(names)} исследований… (несколько минут на каждое)")
        self.log.delete("1.0", "end")
        self._log(f"Вход:  {indir}")
        self._log(f"Выход: {out}")
        self._log(f"Исследований выбрано: {len(names)}\n")

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("FINETUNE_DICOM_TEMP", str(Path(env.get("TEMP", "C:/Temp")) / "ms_review_dicom"))
        cmd = [PYEXE, str(WORKER), "--input", indir, "--output", out,
               "--min-mm3", str(mm), "--select", str(sel_file)]
        if not self.det_var.get():
            cmd.append("--no-detector")
        if not self.cls_var.get():
            cmd.append("--no-classify")
        threading.Thread(target=self._run_worker, args=(cmd, env), daemon=True).start()

    def _run_worker(self, cmd, env):
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not is_noise(line):
                    self.q.put(line)
            self.q.put(("__DONE__", proc.wait()))
        except Exception as e:
            self.q.put(f"[!] Ошибка: {e}")
            self.q.put(("__DONE__", -1))

    def _drain(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item:
                    tag = item[0]
                    if tag == "__DONE__":
                        self._finish(item[1])
                    elif tag == "__STUDIES__":
                        self._populate(item[1])
                else:
                    self._log(str(item))
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _populate(self, items):
        self.checks.set_items(items)
        if items:
            kind = items[0][1]
            self.scan_lbl.config(text=f"найдено: {len(items)}  ({kind})")
        else:
            self.scan_lbl.config(text="исследований не найдено")

    def _finish(self, rc):
        self.pb.stop(); self.run_btn.config(state="normal")
        if rc == 0:
            self.open_btn.config(state="normal")
            self.status.config(text="Готово. Откройте результаты и проверьте разметку в 3D Slicer.",
                               foreground="#177245")
            self._log("\n[OK] Готово. Нажмите «Открыть результаты».")
        else:
            self.status.config(text=f"Завершилось с ошибкой (код {rc}). Смотрите лог выше.",
                               foreground="#a11")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
