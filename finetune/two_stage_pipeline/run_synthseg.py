"""
Run SynthSeg over the reconstructed FLAIR/T2 volumes via Docker.

SynthSeg is contrast-agnostic, so it segments FLAIR/T2 directly (no T1, no
registration). We run it inside a FreeSurfer container to avoid a fragile local
TensorFlow / model-weights install; the training virtual environment is untouched.

Pipeline per study:
1. ``mri_synthseg`` on the reconstructed volume → conformed-space label map;
2. nearest-neighbour resample back onto the original volume grid
   (:func:`~finetune.two_stage_pipeline.synthseg_anatomy.resample_segmentation_to_reference`)
   so ``seg[:, :, z]`` lines up with ``slice_{z}``.

All volumes are batched into a single container call so the model loads once.

    # daemon must be running; image is pulled on first use
    python -m finetune.two_stage_pipeline.run_synthseg
    python -m finetune.two_stage_pipeline.run_synthseg --robust --parc

Cached, aligned segmentations land in
``two_stage_full/anatomy/segmentations/{study}.nii.gz``.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .synthseg_anatomy import resample_segmentation_to_reference


def _find_volumes(volumes_root: Path) -> dict[str, Path]:
    """Map study name -> reconstructed volume path."""
    found: dict[str, Path] = {}
    for study_dir in sorted(p for p in volumes_root.iterdir() if p.is_dir()):
        files = sorted((study_dir / "NIFTI").glob("*.nii*"))
        if files:
            found[study_dir.name] = files[0]
    return found


def build_synthseg_command(
    image: str,
    anatomy_root: Path,
    input_list: Path,
    output_list: Path,
    *,
    robust: bool,
    parc: bool,
    fast: bool,
    cpu: bool,
    threads: int,
) -> list[str]:
    """
    Docker invocation running one batched SynthSeg over a file list.

    The ``cookpa/synthseg`` image's entrypoint is
    ``python .../SynthSeg_predict.py``, so only its flags are passed (no
    ``mri_synthseg``) and no FreeSurfer license is required. ``--i`` / ``--o`` are
    text files listing one container path per line, which tolerates the spaces and
    Cyrillic in study folder names.
    """
    anatomy_abs = anatomy_root.resolve().as_posix()
    command = [
        "docker", "run", "--rm",
        "-v", f"{anatomy_abs}:/anatomy",
        image,
        "--i", f"/anatomy/{input_list.name}",
        "--o", f"/anatomy/{output_list.name}",
        "--threads", str(threads),
    ]
    if cpu:
        command.append("--cpu")
    if robust:
        command.append("--robust")
    if fast:
        command.append("--fast")
    if parc:
        command.append("--parc")
    return command


def run(
    volumes_root: Path,
    anatomy_root: Path,
    *,
    image: str,
    robust: bool,
    parc: bool,
    fast: bool,
    cpu: bool,
    threads: int,
    overwrite: bool = False,
) -> dict[str, object]:
    volumes = _find_volumes(volumes_root)
    if not volumes:
        raise FileNotFoundError(f"No reconstructed volumes under {volumes_root}")

    raw_seg_dir = anatomy_root / "synthseg_raw"
    seg_dir = anatomy_root / "segmentations"
    raw_seg_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: skip studies whose aligned segmentation already exists so a
    # multi-hour batch can be restarted after an interruption.
    if not overwrite:
        already = {s for s in volumes if (seg_dir / f"{s}.nii.gz").is_file()}
        volumes = {s: v for s, v in volumes.items() if s not in already}
        if not volumes:
            return {"volumes": 0, "aligned": 0, "skipped_existing": len(already),
                    "segmentations": str(seg_dir)}

    # File lists use container paths (/anatomy/...), one volume per line.
    input_list = anatomy_root / "synthseg_inputs.txt"
    output_list = anatomy_root / "synthseg_outputs.txt"
    host_pairs: list[tuple[str, Path, Path]] = []
    in_lines, out_lines = [], []
    for study, volume in volumes.items():
        rel_in = volume.resolve().relative_to(anatomy_root.resolve()).as_posix()
        raw_out = raw_seg_dir / f"{study}.nii.gz"
        rel_out = raw_out.resolve().relative_to(anatomy_root.resolve()).as_posix()
        in_lines.append(f"/anatomy/{rel_in}")
        out_lines.append(f"/anatomy/{rel_out}")
        host_pairs.append((study, volume, raw_out))
    input_list.write_text("\n".join(in_lines) + "\n", encoding="utf-8")
    output_list.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    command = build_synthseg_command(
        image, anatomy_root, input_list, output_list,
        robust=robust, parc=parc, fast=fast, cpu=cpu, threads=threads,
    )
    print("running:", " ".join(command))
    subprocess.run(command, check=True)

    # Resample every raw segmentation onto its reference grid for slice alignment.
    aligned = 0
    for study, volume, raw_out in host_pairs:
        if not raw_out.is_file():
            print(f"WARNING: no segmentation produced for {study}")
            continue
        resample_segmentation_to_reference(raw_out, volume, seg_dir / f"{study}.nii.gz")
        aligned += 1
    return {"volumes": len(volumes), "aligned": aligned, "segmentations": str(seg_dir)}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    anatomy_root = (
        project_root / "finetune" / "fileStorage" / "two_stage_full" / "anatomy"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volumes", type=Path, default=anatomy_root / "volumes")
    parser.add_argument("--anatomy-root", type=Path, default=anatomy_root)
    parser.add_argument("--image", default="cookpa/synthseg:conda-0.2")
    parser.add_argument("--robust", action="store_true", help="Higher-quality, slower.")
    parser.add_argument("--fast", action="store_true", help="Faster, lower-res inference.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (GPU busy with training).")
    parser.add_argument("--parc", action="store_true", help="Also cortical parcellation.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Re-segment existing studies.")
    args = parser.parse_args()

    summary = run(
        args.volumes, args.anatomy_root,
        image=args.image, robust=args.robust, parc=args.parc,
        fast=args.fast, cpu=args.cpu, threads=args.threads, overwrite=args.overwrite,
    )
    print(summary)


if __name__ == "__main__":
    main()
