"""
Batch FastSurfer ``--seg_only`` over studies and align the output to FLAIR.

For each study we ensure a T1 exists (via :mod:`lst_ai.reconstruct_t1`), run
FastSurfer's segmentation-only pipeline in the cached ``deepmi/fastsurfer`` image
(no surface reconstruction -> ~1-2 min/subject on GPU), then remap + resample the
result onto the FLAIR grid (:func:`fastsurfer_anatomy.fastsurfer_seg_to_flair`).

The aligned segmentations land in ``anatomy/fastsurfer_segmentations/{study}.nii.gz``
— a drop-in for the SynthSeg directory, so
``train_classifier_anatomy``/``evaluate_seg`` run unchanged with
``--synthseg-dir .../fastsurfer_segmentations``.

    python -m finetune.two_stage_pipeline.run_fastsurfer_seg --studies-file .../val_studies.txt
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

from .fastsurfer_anatomy import fastsurfer_seg_to_flair
from .lst_ai.reconstruct_t1 import reconstruct_t1

IMAGE = "medical-image-segmentation-fastsurfer:latest"
SEG_NAME = "aparc.DKTatlas+aseg.deep.mgz"


def _sid(study: str) -> str:
    """FreeSurfer-safe subject id (no '+', spaces, etc.)."""
    return "s_" + re.sub(r"[^0-9A-Za-z]+", "_", study).strip("_")


def _docker_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _find_flair(nifti_dir: Path) -> Path | None:
    flairs = sorted(nifti_dir.glob("*flair*.nii*"))
    pool = flairs or sorted(nifti_dir.glob("*.nii*"))
    return max(pool, key=lambda p: p.stat().st_size) if pool else None


def run_one(
    study: str,
    input_root: Path,
    volumes_root: Path,
    fs_work: Path,
    seg_out_dir: Path,
    license_dir: Path,
    *,
    overwrite: bool = False,
) -> dict:
    aligned = seg_out_dir / f"{study}.nii.gz"
    if aligned.is_file() and not overwrite:
        return {"study": study, "status": "cached"}

    # 1. T1 (reconstruct if missing)
    t1 = reconstruct_t1(input_root / study, volumes_root)
    if t1 is None:
        return {"study": study, "status": "no_t1"}
    flair = _find_flair(volumes_root / study / "NIFTI")
    if flair is None:
        return {"study": study, "status": "no_flair"}

    sid = _sid(study)
    stage_in = fs_work / "input" / sid
    stage_out = fs_work / "output"
    stage_in.mkdir(parents=True, exist_ok=True)
    stage_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(t1, stage_in / f"{sid}.nii.gz")

    subj_dir = stage_out / sid
    if subj_dir.exists():
        shutil.rmtree(subj_dir, ignore_errors=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{_docker_path(stage_in)}:/data",
        "-v", f"{_docker_path(stage_out)}:/output",
        "-v", f"{_docker_path(license_dir)}:/license",
        IMAGE,
        "/fastsurfer/run_fastsurfer.sh",
        "--t1", f"/data/{sid}.nii.gz",
        "--sid", sid, "--sd", "/output",
        "--seg_only", "--fs_license", "/license/license.txt",
        "--device", "cuda", "--allow_root",
    ]
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**_env()})
    elapsed = round(time.time() - start, 1)
    (fs_work / f"{sid}.log").write_text(
        proc.stdout + "\n=== STDERR ===\n" + proc.stderr, encoding="utf-8", errors="replace"
    )

    mgz = subj_dir / "mri" / SEG_NAME
    if proc.returncode != 0 or not mgz.is_file():
        return {"study": study, "status": "fastsurfer_failed",
                "returncode": proc.returncode, "seconds": elapsed}

    seg_out_dir.mkdir(parents=True, exist_ok=True)
    fastsurfer_seg_to_flair(mgz, flair, aligned)
    # free disk: drop the bulky FastSurfer subject dir, keep only aligned seg
    shutil.rmtree(subj_dir, ignore_errors=True)
    return {"study": study, "status": "ok", "seconds": elapsed, "seg": str(aligned)}


def _env() -> dict:
    import os

    env = dict(os.environ)
    env["MSYS_NO_PATHCONV"] = "1"
    return env


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    base = project_root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "finetune" / "fileStorage" / "input_data")
    parser.add_argument("--volumes", type=Path, default=base / "anatomy" / "volumes")
    parser.add_argument("--fs-work", type=Path, default=base / "fastsurfer")
    parser.add_argument("--seg-out", type=Path, default=base / "anatomy" / "fastsurfer_segmentations")
    parser.add_argument(
        "--license-dir", type=Path,
        default=project_root / "math_app" / "fileStorage" / "fast_surfer_lic",
    )
    parser.add_argument("--studies", nargs="*", default=None)
    parser.add_argument("--studies-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.studies_file:
        studies = [s for s in args.studies_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    elif args.studies:
        studies = args.studies
    else:
        raise SystemExit("Provide --studies or --studies-file")

    ok = 0
    for study in studies:
        result = run_one(
            study, args.input, args.volumes, args.fs_work, args.seg_out,
            args.license_dir, overwrite=args.overwrite,
        )
        print(result, flush=True)
        if result["status"] in ("ok", "cached"):
            ok += 1
    print(f"done: {ok}/{len(studies)} segmentations aligned")


if __name__ == "__main__":
    main()
