# -*- coding: utf-8 -*-
"""Предразметка новых исследований НАШИМ каскадом для проверки врачом.

Детектор 2.5D U-Net (full896, imgsz 896) находит очаги -> анатомический
классификатор (маски SynthSeg + правила врача, 10 признаков) присваивает I/P/S/U ->
на выходе Slicer-готовый 3D labelmap (метки 1..4) + цветовая таблица + инструкция.

Врач открывает {study}_seg.nii.gz в 3D Slicer, убирает лишнее / добавляет
пропущенное / правит класс, сохраняет. Дальше — accept_edits в формат обучения.

    python -m review_tool.preannotate_cascade \
        --volumes finetune/fileStorage/new_training_data/volumes \
        --synthseg-dir finetune/fileStorage/new_training_data/segmentations \
        --output finetune/fileStorage/new_training_data/review_output
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "review_tool"))

from detector_2p5d import Detector2p5d
from seg_nrrd import write_seg_nrrd
from finetune.two_stage_pipeline.predict import AnatomyClassifier
from finetune.two_stage_pipeline.synthseg_anatomy import (
    brain_z_extent, z_norm_from_extent, lesion_features_from_seg)

CLASS_TO_LABEL = {"I": 1, "P": 2, "S": 3, "U": 4}
DEF_DETECTOR = REPO / "finetune/fileStorage/two_stage_full/seg_runs/unet_2p5d_full896/weights/best.pt"
DEF_CLF = REPO / "finetune/fileStorage/two_stage_full/classifier/anatomy_classifier.joblib"

COLORTABLE = (
    "0 background 0 0 0 0\n"
    "1 I_инфратенториальный 255 60 60 255\n"
    "2 P_перивентрикулярный 60 200 60 255\n"
    "3 S_субкортикальный 60 120 255 255\n"
    "4 U_юкстакортикальный 240 200 40 255\n"
)
README = (
    "КАК ПРОВЕРИТЬ ЧЕРНОВИК В 3D SLICER\n"
    "\n"
    "1. File -> Add Data -> выделите СРАЗУ ДВА файла:\n"
    "       {study}_FLAIR.nii.gz     (снимок, фон срезов)\n"
    "       {study}_seg.seg.nrrd     (очаги; имена и цвета I/P/S/U уже внутри)\n"
    "   Ничего не отмечайте в диалоге, просто OK.\n"
    "\n"
    "2. Модуль Segment Editor:\n"
    "       Segmentation  = {study}_seg\n"
    "       Source volume = {study}_FLAIR\n"
    "   В списке появятся 4 сегмента I/P/S/U. Если срезы чёрные — нажмите\n"
    "   в срезовом окне кнопку-булавку (левый верхний угол) и в строке\n"
    "   Background выберите {study}_FLAIR.\n"
    "\n"
    "3. Правьте: Paint — дорисовать, Erase — убрать ложное, класс меняется\n"
    "   выбором нужного сегмента в списке слева.\n"
    "\n"
    "4. Save Data -> перезапишите {study}_seg.seg.nrrd.\n"
    "\n"
    "Файл {study}_seg.nii.gz и MSLesions.ctbl — то же самое в старом формате\n"
    "labelmap (метки 1..4 = I/P/S/U), нужны только для совместимости.\n"
    "\n"
    "Это ЧЕРНОВИК модели: детектор находит не всё и иногда лишнее, класс — подсказка.\n"
    "Финальная разметка — за врачом.\n"
)


def find_flair(study_dir: Path) -> Path | None:
    nifti = study_dir / "NIFTI"
    files = sorted(nifti.glob("*.nii*")) if nifti.is_dir() else []
    return files[0] if files else None


def process_study(study, flair_path, seg_path, detector, clf, outdir, min_voxels):
    img = nib.load(str(flair_path))
    mask3d = detector.mask_volume(flair_path)                 # (H,W,Z) uint8
    seg_vol = np.asarray(nib.load(str(seg_path)).get_fdata()).astype(int)
    if seg_vol.shape != mask3d.shape:
        return {"study": study, "status": "shape_mismatch",
                "seg": seg_vol.shape, "mask": mask3d.shape}
    extent = brain_z_extent(seg_vol)
    labeled, n = ndi.label(mask3d)
    out = np.zeros(mask3d.shape, np.uint8)
    counts = {"I": 0, "P": 0, "S": 0, "U": 0}
    kept = skipped = 0
    for comp in range(1, n + 1):
        mask_c = labeled == comp
        if int(mask_c.sum()) < min_voxels:
            skipped += 1
            continue
        zs = np.nonzero(mask_c.any(axis=(0, 1)))[0]
        zc = int(zs[len(zs) // 2])                            # representative slice
        ys, xs = np.nonzero(mask_c[:, :, zc])
        if xs.size == 0:
            continue
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        zn = z_norm_from_extent(zc, extent)
        feat = lesion_features_from_seg(seg_vol, zc, box, z_norm=zn).vector()
        cls, _prob = clf.classify([feat])[0]
        out[mask_c] = CLASS_TO_LABEL.get(cls, 3)
        counts[cls] = counts.get(cls, 0) + 1
        kept += 1
    sd = outdir / study
    sd.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out.astype(np.uint8), img.affine, img.header),
             str(sd / f"{study}_seg.nii.gz"))
    # то же самое как Slicer-сегментация: открывается сразу редактируемой
    write_seg_nrrd(out, img.affine, sd / f"{study}_seg.seg.nrrd")
    # копия FLAIR рядом (удобно врачу) + служебные файлы
    nib.save(img, str(sd / f"{study}_FLAIR.nii.gz"))
    (sd / "MSLesions.ctbl").write_text(COLORTABLE, encoding="utf-8")
    (sd / "КАК_ПРОВЕРИТЬ_в_Slicer.txt").write_text(README.replace("{study}", study), encoding="utf-8")
    return {"study": study, "status": "ok", "lesions": kept, "skipped_small": skipped, **counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = REPO / "finetune/fileStorage/new_training_data"
    ap.add_argument("--volumes", type=Path, default=base / "volumes")
    ap.add_argument("--synthseg-dir", type=Path, default=base / "segmentations")
    ap.add_argument("--output", type=Path, default=base / "review_output")
    ap.add_argument("--detector-weights", type=Path, default=DEF_DETECTOR)
    ap.add_argument("--classifier", type=Path, default=DEF_CLF)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--min-voxels", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N studies (0 = all).")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    detector = Detector2p5d(weights=args.detector_weights, encoder="resnet34",
                            imgsz=args.imgsz, device=args.device)
    clf = AnatomyClassifier(args.classifier)

    studies = sorted(p for p in args.volumes.iterdir() if p.is_dir())
    if args.limit:
        studies = studies[: args.limit]
    ok = failed = 0
    import csv
    rows = []
    for study_dir in studies:
        study = study_dir.name
        flair = find_flair(study_dir)
        seg = args.synthseg_dir / f"{study}.nii.gz"
        if flair is None or not seg.is_file():
            rows.append({"study": study, "status": "missing_flair_or_seg"}); failed += 1
            print(f"[skip] {study}: FLAIR={flair is not None} seg={seg.is_file()}")
            continue
        try:
            r = process_study(study, flair, seg, detector, clf, args.output, args.min_voxels)
            rows.append(r)
            if r["status"] == "ok":
                ok += 1
                print(f"[ok] {study}: {r['lesions']} lesions  I{r.get('I',0)} P{r.get('P',0)} S{r.get('S',0)} U{r.get('U',0)}")
            else:
                failed += 1; print(f"[warn] {study}: {r['status']}")
        except Exception as exc:  # noqa: BLE001
            rows.append({"study": study, "status": f"error: {str(exc)[:150]}"}); failed += 1
            print(f"[err] {study}: {exc}")
    # отчёт
    keys = ["study", "status", "lesions", "skipped_small", "I", "P", "S", "U", "seg", "mask"]
    with (args.output / "preannotation_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nDONE: ok {ok} / failed {failed} -> {args.output}")


if __name__ == "__main__":
    main()
