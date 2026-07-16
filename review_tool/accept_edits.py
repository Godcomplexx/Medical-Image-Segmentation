# -*- coding: utf-8 -*-
"""
«Принять правки доктора»: проверенные в 3D Slicer сегментации (*_seg.nii.gz)
переводятся обратно в формат обучающей разметки (срез-PNG + YOLO-полигоны класса
1..4) и складываются в review_tool/accepted/. Оттуда их можно влить в обучение —
так замыкается цикл «нашли → врач проверил → в обучение».

Вход : папка с результатами проверки (структура review_tool/output: на каждое
        исследование папка с *_FLAIR.nii.gz и исправленным *_seg.nii.gz), либо
        любая папка, где рядом лежат *_FLAIR.nii.gz и *_seg.nii.gz.
Выход: accepted/{images,labels}/slice_{z}_{study}.png|.txt + manifest.csv
"""
import argparse, sys, re
from pathlib import Path
import numpy as np
import cv2
import nibabel as nib

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from finetune.src.yolo.image_generator import normalize_mri_volume

CLASS_NAMES = {1: "I", 2: "P", 3: "S", 4: "U"}
CAND = 5


def find_pairs(indir: Path):
    """Находит пары (study, flair_path, seg_path)."""
    pairs = []
    for seg in sorted(indir.rglob("*_seg.nii.gz")):
        stem = seg.name[:-len("_seg.nii.gz")]
        flair = seg.with_name(f"{stem}_FLAIR.nii.gz")
        if not flair.is_file():
            cand = list(seg.parent.glob("*_FLAIR.nii.gz"))
            flair = cand[0] if cand else None
        if flair is not None:
            pairs.append((stem, flair, seg))
    return pairs


def polygons_from_slice(mask2d, cls, min_area):
    """Контуры очагов класса cls на срезе -> список полигонов (пиксели)."""
    m = (mask2d == cls).astype(np.uint8)
    if not m.any():
        return []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        pts = c.reshape(-1, 2)
        if len(pts) < 3:
            continue
        out.append(pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "output"))
    ap.add_argument("--output", default=str(ROOT / "accepted"))
    ap.add_argument("--min-area", type=float, default=6.0, help="мин. площадь очага, px²")
    args = ap.parse_args()

    indir, outdir = Path(args.input), Path(args.output)
    (outdir / "images").mkdir(parents=True, exist_ok=True)
    (outdir / "labels").mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(indir)
    if not pairs:
        print(f"[!] В '{indir}' не найдено пар *_FLAIR.nii.gz + *_seg.nii.gz.")
        print("    Укажите папку с результатами проверки (review_tool/output).")
        return 1
    print(f"[i] Исследований к приёму: {len(pairs)}")

    # val-исследования (для предупреждения об утечке)
    val_names = set()
    val_dir = REPO / "finetune" / "fileStorage" / "train_set" / "labels" / "val"
    if val_dir.is_dir():
        for lp in val_dir.glob("*.txt"):
            m = re.match(r"^slice_\d+_(.+)$", lp.stem)
            if m:
                val_names.add(m.group(1))

    manifest = ["study,slices_with_lesions,lesions,in_val_split"]
    total_les = 0
    for stem, flair, seg in pairs:
        F = normalize_mri_volume(nib.load(str(flair)).get_fdata())      # HWZ uint8
        L = np.asarray(nib.load(str(seg)).get_fdata()).astype(np.uint8)  # HWZ 0..5
        if F.shape != L.shape:
            print(f"    [пропуск] {stem}: размеры FLAIR и разметки не совпадают {F.shape} vs {L.shape}")
            continue
        cand_left = int((L == CAND).sum() > 0)
        if cand_left:
            print(f"    [!] {stem}: остались неразобранные кандидаты «?» — им класс НЕ присвоен, "
                  f"они пропущены. Классифицируйте их в Slicer и примите снова.")
        H, W, Z = F.shape
        slices_used = les_cnt = 0
        in_val = stem in val_names
        for z in range(Z):
            lines = []
            for cls in CLASS_NAMES:
                for pts in polygons_from_slice(L[:, :, z], cls, args.min_area):
                    coords = []
                    for x, y in pts:
                        coords.append(f"{x / W:.6f}")
                        coords.append(f"{y / H:.6f}")
                    lines.append(str(cls) + " " + " ".join(coords))
                    les_cnt += 1
            if not lines:
                continue
            name = f"slice_{z}_{stem}"
            cv2.imencode(".png", F[:, :, z])[1].tofile(str(outdir / "images" / f"{name}.png"))
            (outdir / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            slices_used += 1
        total_les += les_cnt
        manifest.append(f'"{stem}",{slices_used},{les_cnt},{int(in_val)}')
        flag = "  [в VAL — не смешивать с train!]" if in_val else ""
        print(f"    {stem}: срезов {slices_used}, очагов {les_cnt}{flag}")

    (outdir / "manifest.csv").write_text("\n".join(manifest), encoding="utf-8")
    print(f"\n[OK] Готово. Всего очагов принято: {total_les}")
    print(f"[OK] Разметка: {outdir}\\labels , срезы: {outdir}\\images")
    print("[OK] Как влить в обучение: перенести srez-PNG в train_set/images/finetune")
    print("     и .txt в train_set/labels/finetune (исследования из VAL — держать отдельно),")
    print("     затем переобучить (напр. nnU-Net) на обновлённой разметке.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
