"""
Complete the (sparse) detector annotation with LST-AI pseudo-lesions.

Our binary detector labels mark only the lesions the annotators drew; LST-AI (a
strong pretrained detector) finds many *real* lesions they missed (see the
unmarked-lesion montage). Training a detector on the sparse labels caps its recall
and teaches it to ignore real lesions. Here we distil LST-AI into the training
labels: every LST-AI lesion component that does **not** overlap an existing label
is added as a new class-0 polygon, so the student detector learns the full lesion
set from our own data.

Robustness:
* LST-AI output is nearest-neighbour resized to the PNG grid (z already matches).
* Studies whose LST-AI volume does not span the annotation z-range (reconstructed
  from a different acquisition) keep their original labels unchanged.
* Tiny components (< ``min_area`` px) are dropped to limit artefacts.

    python -m finetune.two_stage_pipeline.lst_ai.merge_pseudolabels \
        --labels .../detector_2p5d/labels/train --out .../labels_completed/train
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import nibabel as nib
from PIL import Image

_STEM = re.compile(r"^slice_(\d+)_(.+)$")
_SEG = "space-flair_seg-lst.nii.gz"


def _polys_to_mask(label_path: Path, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    if not label_path.is_file():
        return mask
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = list(map(float, parts[1:]))
        pts = np.array(
            [(int(coords[i] * width), int(coords[i + 1] * height)) for i in range(0, len(coords) - 1, 2)],
            np.int32,
        )
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask


def _component_to_poly(comp_mask: np.ndarray, width: int, height: int) -> str | None:
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        x, y, w, h = cv2.boundingRect(cnt)
        approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    coords = " ".join(f"{px / width:.6f} {py / height:.6f}" for px, py in approx)
    return f"0 {coords}"


def merge(
    labels_dir: Path,
    images_dir: Path,
    lst_root: Path,
    out_dir: Path,
    *,
    min_area: int = 10,
    overlap_frac: float = 0.2,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    binary: dict[str, np.ndarray | None] = {}
    max_z: dict[str, int] = {}
    label_files = sorted(labels_dir.glob("*.txt"))
    for lp in label_files:
        m = _STEM.match(lp.stem)
        if m:
            max_z[m.group(2)] = max(max_z.get(m.group(2), 0), int(m.group(1)))

    def load_binary(study: str):
        if study not in binary:
            p = lst_root / study / "output" / _SEG
            binary[study] = np.asarray(nib.load(str(p)).get_fdata()).astype(np.uint8) if p.is_file() else None
        return binary[study]

    stats = {"files": 0, "added": 0, "skipped_studies": set(), "studies_completed": set()}
    for lp in label_files:
        m = _STEM.match(lp.stem)
        original = lp.read_text() if lp.is_file() else ""
        if m is None:
            (out_dir / lp.name).write_text(original)
            continue
        z, study = int(m.group(1)), m.group(2)
        b = load_binary(study)
        covered = b is not None and b.shape[2] > max_z.get(study, 0)
        if not covered:
            if b is not None:
                stats["skipped_studies"].add(study)
            (out_dir / lp.name).write_text(original)  # keep original labels
            continue

        image_path = images_dir / f"{lp.stem}.png"
        if not image_path.is_file():
            (out_dir / lp.name).write_text(original)
            continue
        with Image.open(image_path) as im:
            W, H = im.width, im.height
        plane = b[:, :, z] if 0 <= z < b.shape[2] else np.zeros((b.shape[0], b.shape[1]), np.uint8)
        if plane.shape != (H, W):
            plane = cv2.resize(plane, (W, H), interpolation=cv2.INTER_NEAREST)

        existing = _polys_to_mask(lp, W, H)
        count, comp = cv2.connectedComponents(plane.astype(np.uint8), connectivity=8)
        new_lines = []
        for c in range(1, count):
            cm = (comp == c).astype(np.uint8)
            area = int(cm.sum())
            if area < min_area:
                continue
            overlap = int((cm.astype(bool) & existing.astype(bool)).sum())
            if overlap >= overlap_frac * area:
                continue  # already annotated
            line = _component_to_poly(cm, W, H)
            if line:
                new_lines.append(line)
        merged = original.rstrip("\n")
        if new_lines:
            merged = (merged + "\n" if merged else "") + "\n".join(new_lines)
            stats["added"] += len(new_lines)
            stats["studies_completed"].add(study)
        (out_dir / lp.name).write_text(merged + ("\n" if merged else ""))
        stats["files"] += 1

    return {
        "files": stats["files"],
        "pseudolabels_added": stats["added"],
        "studies_completed": len(stats["studies_completed"]),
        "studies_skipped_zmismatch": len(stats["skipped_studies"]),
        "out": str(out_dir),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    base = root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=base / "detector_2p5d" / "labels" / "train")
    parser.add_argument("--images", type=Path, default=base / "detector_2p5d" / "images" / "train")
    parser.add_argument("--lst-root", type=Path, default=base / "lst_ai")
    parser.add_argument("--out", type=Path, default=base / "detector_2p5d" / "labels_completed" / "train")
    parser.add_argument("--min-area", type=int, default=10)
    args = parser.parse_args()
    print(merge(args.labels, args.images, args.lst_root, args.out, min_area=args.min_area))


if __name__ == "__main__":
    main()
