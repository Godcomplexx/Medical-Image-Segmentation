"""
Build a 2.5D detector dataset: stack neighbouring slices as RGB channels.

Each grayscale axial slice ``slice_{z}_{study}`` is replaced by an RGB image whose
channels are the previous / current / next slices ``(z-1, z, z+1)`` of the same
study. Adjacent slices give the detector volumetric context, which is the standard
remedy for missed small MS lesions on single 2D slices. The current channel is
pixel-identical to the original slice, so the existing YOLO labels stay valid.

Neighbours come from the reconstructed volumes
(``anatomy/volumes/{study}/NIFTI``); slices at a volume boundary or from a study
without a reconstructed volume fall back to replicating the current slice.

    python -m finetune.two_stage_pipeline.build_2p5d

Output: ``two_stage_full/detector_2p5d/{images,labels}/{train,val}`` + ``dataset.yaml``.
Train the detector on it with ``--data .../detector_2p5d/dataset.yaml``.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from finetune.src.yolo.image_generator import normalize_mri_volume
from .anatomy import parse_study_slice

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _load_normalized_volume(volumes_root: Path, study: str) -> np.ndarray | None:
    import nibabel as nib

    nifti_dir = volumes_root / study / "NIFTI"
    if not nifti_dir.is_dir():
        return None
    files = sorted(nifti_dir.glob("*.nii*"))
    if not files:
        return None
    return normalize_mri_volume(nib.load(str(files[0])).get_fdata())


def _stack_2p5d(volume: np.ndarray, z: int, fallback: np.ndarray) -> np.ndarray:
    """RGB = (z-1, z, z+1) with edge clamping; ``fallback`` if volume unusable."""
    depth = volume.shape[2]
    if not 0 <= z < depth or volume[:, :, z].shape != fallback.shape:
        return np.repeat(fallback[:, :, None], 3, axis=2)
    prev_z = max(z - 1, 0)
    next_z = min(z + 1, depth - 1)
    return np.stack(
        [volume[:, :, prev_z], volume[:, :, z], volume[:, :, next_z]], axis=2
    ).astype(np.uint8)


def build(detector_root: Path, volumes_root: Path, output_root: Path) -> dict[str, int]:
    counts = {"stacked": 0, "fallback": 0}
    for split in ("train", "val"):
        image_dir = detector_root / "images" / split
        label_dir = detector_root / "labels" / split
        out_images = output_root / "images" / split
        out_labels = output_root / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        images = sorted(
            p for p in image_dir.iterdir() if p.suffix.casefold() in IMAGE_EXTENSIONS
        )
        # Sort by study so each volume is loaded once and evicted on change.
        images.sort(key=lambda p: parse_study_slice(p.name)[0])
        current_study: str | None = None
        volume: np.ndarray | None = None
        for image_path in images:
            study, z = parse_study_slice(image_path.name)
            if study != current_study:
                volume = _load_normalized_volume(volumes_root, study)
                current_study = study
            current = np.asarray(Image.open(image_path).convert("L"))
            if volume is not None:
                rgb = _stack_2p5d(volume, z, current)
                counts["stacked" if volume[:, :, z].shape == current.shape else "fallback"] += 1
            else:
                rgb = np.repeat(current[:, :, None], 3, axis=2)
                counts["fallback"] += 1
            Image.fromarray(rgb, mode="RGB").save(out_images / f"{image_path.stem}.png")

            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                _link_or_copy(label_path, out_labels / label_path.name)
            else:
                (out_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")

    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: lesion\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    root = project_root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, default=root / "detector")
    parser.add_argument("--volumes", type=Path, default=root / "anatomy" / "volumes")
    parser.add_argument("--output", type=Path, default=root / "detector_2p5d")
    args = parser.parse_args()

    counts = build(args.detector, args.volumes, args.output)
    print({"output": str(args.output), **counts})


if __name__ == "__main__":
    main()
