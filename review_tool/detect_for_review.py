# -*- coding: utf-8 -*-
"""
Автоматическая генерация КАНДИДАТ-разметки очагов РС для проверки в 3D Slicer.

Готовое решение (обученный nnU-Net 3d_fullres, только FLAIR) находит очаги и
сразу предлагает анатомический класс I/P/S/U. Врач открывает результат в 3D Slicer,
проверяет, правит класс и дорисовывает пропущенное — это в разы быстрее разметки
с нуля и даёт полную согласованную разметку для дообучения.

Вход (любой из двух — определяется автоматически):
  • папка с DICOM-исследованиями (подпапка на исследование, внутри DICOM/…),
    FLAIR-серия извлекается автоматически;
  • папка с готовыми FLAIR-объёмами (*.nii / *.nii.gz).
Выход: на каждое исследование — папка с FLAIR + сегментацией (метки 0..4),
       цветовой таблицей Slicer и инструкцией.

Классы: 1=I (инфратенториальный), 2=P (перивентрикулярный),
        3=S (субкортикальный), 4=U (юкстакортикальный).
"""
import argparse, os, re, sys, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
ENV  = REPO / "finetune" / "fileStorage" / "nnunet_env"

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ASCII temp для GDCM (кириллица в путях ломает конвертацию на Windows)
os.environ.setdefault("FINETUNE_DICOM_TEMP",
                      str(Path(tempfile.gettempdir()) / "ms_review_dicom"))
try:
    import pydicom
    from finetune.src.filetools.dicom_to_nifti_converter import (
        process_dicom_to_nifti, t2_filter)
    HAVE_DICOM = True
except Exception:
    HAVE_DICOM = False

CLASS_NAMES = {1: "I", 2: "P", 3: "S", 4: "U"}
CLASS_FULL  = {1: "инфратенториальный", 2: "перивентрикулярный",
               3: "субкортикальный", 4: "юкстакортикальный"}
CLASS_RGB = {1: (255, 60, 60), 2: (60, 200, 60), 3: (60, 120, 255), 4: (255, 210, 40)}
CAND = 5  # очаг найден 2-м детектором, но класс не определён — доктор присваивает
CAND_NAME = "кандидат_без_класса"
CAND_RGB = (255, 0, 255)

COLORTABLE = "\n".join(
    ["# 3D Slicer color table — очаги РС (I/P/S/U + кандидаты)", "0 Фон 0 0 0 0"] +
    [f"{k} {CLASS_NAMES[k]}_{CLASS_FULL[k]} {r} {g} {b} 255"
     for k, (r, g, b) in CLASS_RGB.items()] +
    [f"{CAND} {CAND_NAME} {CAND_RGB[0]} {CAND_RGB[1]} {CAND_RGB[2]} 255"]) + "\n"

README = """КАК ОТКРЫТЬ И ПРОВЕРИТЬ РАЗМЕТКУ В 3D SLICER
=================================================

Для каждого исследования в его папке лежат:
  *_FLAIR.nii.gz   — снимок FLAIR (фон)
  *_seg.nii.gz     — предложенная разметка очагов (метки 1..4 = I/P/S/U)
  MSLesions.ctbl   — цветовая таблица (имена и цвета классов)

ШАГИ:
1. Откройте 3D Slicer.
2. Перетащите в окно Slicer файл *_FLAIR.nii.gz  (загрузится как объём — фон).
3. Перетащите файл *_seg.nii.gz. В диалоге загрузки поставьте галочку
   "LabelMap" (Slicer обычно определяет автоматически). Нажмите OK.
4. Загрузите цвета: меню File → Add Data → выберите MSLesions.ctbl.
5. Модуль Data: у загруженного *_seg установите Color node = MSLesions.
6. Правый клик на *_seg → "Convert labelmap to segmentation".
   Теперь очаги стали редактируемыми сегментами с именами I/P/S/U.
7. Модуль "Segment Editor": проверьте каждый очаг.
   - неверный класс: перенесите воксели в правильный сегмент;
   - лишний очаг: сотрите (Erase);
   - ПРОПУЩЕННЫЙ очаг: дорисуйте кистью (Paint) в нужный сегмент.
8. Сохраните: Save Data → *_seg.nii.gz (перезапишите) ИЛИ .seg.nrrd.

ВАЖНО: разметка НЕ финальная — это предложение модели. Слабое место модели —
класс S (часть очагов пропускает) и редкий класс I. Особое внимание им.

ОТКУДА ОЧАГИ: часть находит объёмная модель (nnU-Net) и сразу присваивает класс;
часть добавляет второй, 2.5D-детектор (ловит то, что объёмная пропустила) — этим
очагам класс присваивается автоматически (предположительно). ВСЕ очаги нужно
проверить; особое внимание — редкому I и частому S.

МЕТКА 5 (кандидат_без_класса, малиновый) появляется только если авто-класс
отключён — тогда присвойте класс вручную (перенесите в сегмент I/P/S/U) или сотрите.

Проверенные и исправленные файлы возвращаются в обучение (батник «Принять_правки»).
"""


# ----------------------------- сбор исследований -----------------------------
_NON_DICOM_EXT = {
    ".json", ".mrk", ".nii", ".gz", ".txt", ".csv",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".xml", ".log",
}


def looks_like_dicom_study(folder: Path) -> bool:
    """Папка считается DICOM-исследованием, если содержит DICOM-файлы.

    Заведомо не-DICOM файлы (разметка Markups .json/.mrk, NIfTI, картинки)
    пропускаются, чтобы они не «съедали» бюджет проверки — иначе исследования
    с большим числом файлов разметки перед DICOM ошибочно считались не-DICOM.
    """
    if (folder / "DICOM").is_dir():
        return True
    if not HAVE_DICOM:
        return False
    checked = 0
    for f in folder.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() in _NON_DICOM_EXT:
            continue
        try:
            pydicom.dcmread(str(f), stop_before_pixels=True)
            return True
        except Exception:
            checked += 1
            if checked >= 2000:
                break
    return False


def pick_flair(nifti_dir: Path):
    files = list(nifti_dir.glob("*.nii*"))
    if not files:
        return None
    flair = [f for f in files if re.search(r"flair|fluid", f.name, re.I)]
    if flair:
        return flair[0]
    # иначе — серия с наименьшим размером в плоскости (FLAIR обычно 256–288, T2/DUAL — 512+)
    def inplane(f):
        try:
            return nib.load(str(f)).shape[0]
        except Exception:
            return 10 ** 9
    return sorted(files, key=inplane)[0]


def convert_dicom_study(study: Path, out_dir: Path):
    """DICOM-исследование -> FLAIR NIfTI. Возвращает путь к FLAIR или None."""
    src = study / "DICOM" if (study / "DICOM").is_dir() else study
    nifti_dir = out_dir / "NIFTI"
    try:
        process_dicom_to_nifti(dicom_source=src, dicom_temp_dir=out_dir / "_tmp",
                               nifti_output_dir=nifti_dir, filter_func=t2_filter)
    except Exception as e:
        print(f"    [конвертация не удалась] {study.name}: {e}")
        return None
    return pick_flair(nifti_dir)


def orig_stem(path: Path):
    n = path.name
    for ext in (".nii.gz", ".nii"):
        if n.endswith(ext):
            return n[:-len(ext)]
    return path.stem


def scan_studies(indir: Path):
    """Быстрый список исследований (без конвертации): [(name, kind)]."""
    indir = Path(indir)
    if not indir.is_dir():
        return []
    dicom_dirs = [d for d in sorted(indir.iterdir())
                  if d.is_dir() and looks_like_dicom_study(d)]
    if dicom_dirs:
        return [(d.name, "DICOM") for d in dicom_dirs]
    niis = []
    for pat in ("*.nii", "*.nii.gz"):
        niis += list(indir.rglob(pat))
    return [(orig_stem(f), "NIfTI") for f in sorted(set(niis))]


def gather_studies(indir: Path, work: Path, allowed=None):
    """Возвращает список (study_id, flair_path). Определяет DICOM/NIfTI автоматически.

    ``allowed`` — множество имён исследований для обработки (None = все)."""
    studies = []
    # 1) DICOM-режим: подпапки-исследования
    dicom_dirs = [d for d in sorted(indir.iterdir()) if d.is_dir()
                  and looks_like_dicom_study(d)] if indir.is_dir() else []
    if allowed is not None:
        dicom_dirs = [d for d in dicom_dirs if d.name in allowed]
    if dicom_dirs:
        print(f"[i] Режим DICOM: к обработке исследований — {len(dicom_dirs)}")
        for idx, d in enumerate(dicom_dirs):
            print(f"    конвертация FLAIR: {d.name}")
            # Префикс с индексом гарантирует уникальный каталог даже при совпадении
            # первых символов длинных имён (иначе повторная конвертация затирала FLAIR соседа).
            conv = work / f"{idx:04d}_{re.sub(r'[^0-9A-Za-z]+', '_', d.name)[:36]}"
            flair = convert_dicom_study(d, conv)
            if flair is not None and Path(flair).exists():
                studies.append((d.name, flair))
            else:
                print(f"    [пропуск] FLAIR не извлечён: {d.name}")
        return studies
    # 2) NIfTI-режим: готовые объёмы
    niis = []
    for pat in ("*.nii", "*.nii.gz"):
        niis += list(indir.rglob(pat))
    niis = sorted(set(niis))
    if allowed is not None:
        niis = [f for f in niis if orig_stem(f) in allowed]
    if niis:
        print(f"[i] Режим NIfTI: к обработке объёмов — {len(niis)}")
        for f in niis:
            studies.append((orig_stem(f), f))
    return studies


# ----------------------------- пост-обработка --------------------------------
def size_filter(lab, zooms, min_mm3):
    if min_mm3 <= 0 or not HAVE_SCIPY:
        return lab, 0
    vox = float(abs(zooms[0] * zooms[1] * zooms[2]))
    min_vox = max(1, int(round(min_mm3 / max(vox, 1e-6))))
    out, removed = lab.copy(), 0
    for c in (1, 2, 3, 4, CAND):
        m = (lab == c)
        if not m.any():
            continue
        cc, n = ndi.label(m)
        if n == 0:
            continue
        sizes = ndi.sum(np.ones_like(cc), cc, index=np.arange(1, n + 1))
        for i, sz in enumerate(sizes, 1):
            if sz < min_vox:
                out[cc == i] = 0
                removed += 1
    return out, removed


def count_lesions(lab):
    res = {}
    for c, name in list(CLASS_NAMES.items()) + [(CAND, "?")]:
        m = (lab == c)
        res[name] = int(ndi.label(m)[1]) if (HAVE_SCIPY and m.any()) else int(m.any())
    return res


def ascii_id(stem, idx):
    s = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_")[:32]
    return f"case{idx:03d}_{s}" if s else f"case{idx:03d}"


# --------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(ROOT / "input"))
    ap.add_argument("--output", default=str(ROOT / "output"))
    ap.add_argument("--min-mm3", type=float, default=15.0,
                    help="мин. объём очага (мм^3); ~15 ≈ очаг 3 мм. 0 = не фильтровать")
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--select", default=None,
                    help="файл со списком имён исследований (по одному в строке) для обработки")
    ap.add_argument("--no-detector", action="store_true",
                    help="не дополнять находки 2.5D-детектором (только nnU-Net)")
    ap.add_argument("--no-classify", action="store_true",
                    help="не присваивать класс кандидатам 2.5D (оставлять «?»)")
    ap.add_argument("--detector-weights", default=None)
    ap.add_argument("--detector-imgsz", type=int, default=896,
                    help="разрешение входа 2.5D-детектора; должно совпадать с обучением")
    args = ap.parse_args()

    indir, outdir = Path(args.input), Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "MSLesions.ctbl").write_text(COLORTABLE, encoding="utf-8")
    (outdir / "КАК_ПРОВЕРИТЬ_в_Slicer.txt").write_text(README, encoding="utf-8")

    allowed = None
    if args.select:
        allowed = set(x.strip() for x in Path(args.select).read_text(encoding="utf-8").splitlines() if x.strip())

    conv_root = Path(tempfile.mkdtemp(prefix="ms_conv_"))
    studies = gather_studies(indir, conv_root, allowed)
    if not studies:
        print(f"[!] В '{indir}' не найдено ни DICOM-исследований, ни FLAIR-объёмов.")
        print(f"    Положите туда папки DICOM или файлы .nii/.nii.gz и запустите снова.")
        if not HAVE_DICOM:
            print("[!] DICOM-режим недоступен (нет dicom2nifti/pydicom в окружении).")
        shutil.rmtree(conv_root, ignore_errors=True)
        return 1

    os.environ["nnUNet_raw"]          = str(ENV / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(ENV / "nnUNet_preprocessed")
    os.environ["nnUNet_results"]      = str(ENV / "nnUNet_results")
    predict_exe = Path(sys.executable).parent / "nnUNetv2_predict.exe"
    if not predict_exe.is_file():
        predict_exe = Path(sys.executable).parent / "nnUNetv2_predict"

    stage_in  = Path(tempfile.mkdtemp(prefix="nnraw_in_"))
    stage_out = Path(tempfile.mkdtemp(prefix="nnraw_out_"))
    id_map, manifest = {}, []
    for i, (sid, flair) in enumerate(studies):
        if flair is None or not Path(flair).exists():
            print(f"    [пропуск] FLAIR отсутствует, исследование не размечается: {sid}")
            continue
        cid = ascii_id(sid, i)
        # nnU-Net ждёт РОВНО одну модальность: на 4-мерном объёме его препроцессор
        # падает (normalization_schemes[c] -> IndexError), воркеры умирают и вместе
        # с ними теряется ВЕСЬ пакет. Поэтому проверяем форму до запуска.
        staged = stage_in / f"{cid}_0000.nii.gz"
        try:
            img4 = nib.load(str(flair))
            if len(img4.shape) > 3:
                extra = [d for d in img4.shape[3:] if d > 1]
                if extra:
                    print(f"    [пропуск] объём {img4.shape} — не одна модальность: {sid}")
                    continue
                arr3 = np.asarray(img4.dataobj).reshape(img4.shape[:3])
                nib.save(nib.Nifti1Image(arr3, img4.affine, img4.header), str(staged))
            else:
                shutil.copy2(flair, staged)
        except Exception as e:  # noqa: BLE001
            print(f"    [пропуск] не читается FLAIR ({e}): {sid}")
            continue
        id_map[cid] = (sid, flair)
        manifest.append((cid, sid, str(flair)))
    if not id_map:
        print("[!] Ни одного FLAIR не удалось подготовить — размечать нечего.")
        shutil.rmtree(conv_root, ignore_errors=True)
        return 1
    (outdir / "manifest.csv").write_text(
        "case_id,study,source_flair\n" +
        "\n".join(f'{c},"{s}","{f}"' for c, s, f in manifest), encoding="utf-8")

    print(f"\n[i] Запуск детектора (nnU-Net 3d_fullres, {args.checkpoint})...")
    cmd = [str(predict_exe), "-i", str(stage_in), "-o", str(stage_out),
           "-d", "1", "-c", "3d_fullres", "-f", "0", "-chk", args.checkpoint]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[!] Ошибка запуска nnU-Net. Проверьте модель и окружение.")
        return r.returncode

    # 2.5D-детектор: дополняет находки nnU-Net (union), помечает новые как «?» (метка 5)
    detector = None
    if not args.no_detector:
        try:
            from detector_2p5d import Detector2p5d, DEFAULT_WEIGHTS
            # запасной путь: прогон, на котором детектор обучен сейчас (full896)
            fallback = (REPO / "finetune" / "fileStorage" / "two_stage_full" /
                        "seg_runs" / "unet_2p5d_full896" / "weights" / "best.pt")
            w = args.detector_weights or (DEFAULT_WEIGHTS if Path(DEFAULT_WEIGHTS).is_file()
                                          else fallback)
            if Path(w).is_file():
                print(f"[i] Загрузка 2.5D-детектора (дополнение находок): "
                      f"{Path(w).parent.parent.name}, imgsz={args.detector_imgsz}")
                detector = Detector2p5d(weights=w, imgsz=args.detector_imgsz)
            else:
                print("[i] 2.5D-детектор пропущен (нет весов).")
        except Exception as e:
            print(f"[i] 2.5D-детектор недоступен ({e}); используется только nnU-Net.")

    anat = None
    if detector is not None and not args.no_classify:
        try:
            from anatomy_classify import AnatomyClassifier
            anat = AnatomyClassifier()
            print("[i] Авто-класс кандидатов: " +
                  ("обученный классификатор" if anat.model is not None else "клиническое правило"))
        except Exception as e:
            print(f"[i] Авто-класс недоступен ({e}); кандидаты помечаются «?».")

    print("\n[i] Пост-обработка и раскладка для Slicer...")
    total = {n: 0 for n in list(CLASS_NAMES.values()) + ["?"]}
    for cid, (sid, flair) in id_map.items():
        pred = stage_out / f"{cid}.nii.gz"
        if not pred.is_file():
            print(f"    [пропуск] нет предсказания: {sid}")
            continue
        img = nib.load(str(pred))
        lab = np.asarray(img.get_fdata()).astype(np.uint8)
        # union со 2.5D-детектором: его находки вне очагов nnU-Net -> метка «?»
        if detector is not None:
            try:
                det = detector.mask_volume(flair)
                gray = getattr(detector, "last_gray", None)
                if det.shape == lab.shape:
                    Zt = lab.shape[2]
                    for z in range(Zt):
                        dz = det[:, :, z]
                        if not dz.any():
                            continue
                        ncc, cc = cv2.connectedComponents(dz.astype(np.uint8), 8)
                        lz = lab[:, :, z]
                        for ci in range(1, ncc):
                            comp = (cc == ci)
                            if (lz[comp] > 0).any():
                                continue  # уже найдено nnU-Net — сохраняем его класс
                            if anat is not None and gray is not None:
                                ys, xs = np.nonzero(comp)
                                box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                                lz[comp] = anat.classify(gray[:, :, z], box,
                                                         z / max(Zt - 1, 1), (sid, z))
                            else:
                                lz[comp] = CAND
                        lab[:, :, z] = lz
            except Exception as e:
                print(f"    [2.5D пропущен для {sid}: {e}]")
        lab, removed = size_filter(lab, img.header.get_zooms()[:3], args.min_mm3)
        counts = count_lesions(lab)
        for k, v in counts.items():
            total[k] += v
        safe = re.sub(r'[\\/:*?"<>|]+', "_", sid).strip() or cid
        sd = outdir / safe
        sd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(flair, sd / f"{safe}_FLAIR.nii.gz")
        nib.save(nib.Nifti1Image(lab, img.affine, img.header), str(sd / f"{safe}_seg.nii.gz"))
        shutil.copy2(outdir / "MSLesions.ctbl", sd / "MSLesions.ctbl")
        print(f"    {safe}: I={counts['I']} P={counts['P']} S={counts['S']} U={counts['U']} ?={counts['?']}"
              + (f"  (мелких отфильтровано: {removed})" if removed else ""))

    for d in (stage_in, stage_out, conv_root):
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n[OK] Готово. Всего: I={total['I']} P={total['P']} S={total['S']} U={total['U']} ?(кандидаты)={total['?']}")
    print(f"[OK] Результаты: {outdir}")
    print(f"[OK] Инструкция:  {outdir / 'КАК_ПРОВЕРИТЬ_в_Slicer.txt'}")
    if not HAVE_SCIPY:
        print("[!] scipy не найден — фильтр мелких очагов пропущен.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
