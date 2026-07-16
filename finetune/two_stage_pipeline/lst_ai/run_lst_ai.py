"""
Run the LST-AI Docker container on reconstructed T1 + FLAIR volumes.

LST-AI (``jqmcginnis/lst-ai:v1.2.0``) segments MS lesions and annotates each by
McDonald zone in one pass, entirely in FLAIR native space. It skull-strips and
registers T1->FLAIR internally, so we only feed the two raw volumes.

For each study we stage an ``input`` dir holding ``t1.nii.gz`` + ``flair.nii.gz``,
mount input/output/temp into the container, and run it. Outputs land in
``lst_ai/{study}/output``:

* ``space-flair_seg-lst.nii.gz``            -- binary lesion mask (FLAIR space)
* ``space-flair_desc-annotated_seg-lst.nii.gz`` -- zone labels 1=P 2=U 3=S 4=I
* ``annotated_lesion_stats.csv``            -- per-zone lesion counts/volume

``--device cpu`` avoids competing with GPU training (the RTX 3060 Ti has little
free VRAM while the U-Net trains); pass ``--device 0`` once the GPU is free.

    python -m finetune.two_stage_pipeline.lst_ai.run_lst_ai --studies +9 +219 --device cpu
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

IMAGE = "jqmcginnis/lst-ai:v1.2.0"
ANNOTATED = "space-flair_desc-annotated_seg-lst.nii.gz"
SEG = "space-flair_seg-lst.nii.gz"


def _largest(files: list[Path]) -> Path | None:
    files = [f for f in files if f.is_file()]
    return max(files, key=lambda p: p.stat().st_size) if files else None


def _find_flair(nifti_dir: Path) -> Path | None:
    return _largest(sorted(nifti_dir.glob("*flair*.nii*"))) or _largest(
        sorted(nifti_dir.glob("*.nii*"))
    )


def _docker_path(path: Path) -> str:
    """Windows absolute path -> forward-slash form Docker Desktop accepts in -v."""
    return str(path.resolve()).replace("\\", "/")


def run_one(
    study: str,
    volumes_root: Path,
    lst_root: Path,
    *,
    device: str = "cpu",
    threshold: float = 0.5,
    overwrite: bool = False,
) -> dict:
    vol_dir = volumes_root / study
    flair = _find_flair(vol_dir / "NIFTI")
    t1 = _largest(sorted((vol_dir / "NIFTI_T1").glob("*.nii*")))
    if flair is None or t1 is None:
        return {"study": study, "status": "missing_input",
                "t1": bool(t1), "flair": bool(flair)}

    work = lst_root / study
    out_dir = work / "output"
    annotated = out_dir / ANNOTATED
    if annotated.is_file() and not overwrite:
        return {"study": study, "status": "cached", "annotated": str(annotated)}

    in_dir, temp_dir = work / "input", work / "temp"
    for d in (in_dir, out_dir, temp_dir):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(t1, in_dir / "t1.nii.gz")
    shutil.copy2(flair, in_dir / "flair.nii.gz")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{_docker_path(in_dir)}:/in",
        "-v", f"{_docker_path(out_dir)}:/out",
        "-v", f"{_docker_path(temp_dir)}:/tmp2",
    ]
    if device != "cpu":
        cmd += ["--gpus", "all"]
    cmd += [
        IMAGE,
        "--t1", "/in/t1.nii.gz",
        "--flair", "/in/flair.nii.gz",
        "--output", "/out",
        "--temp", "/tmp2",
        "--device", device,
        "--threshold", str(threshold),
    ]

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (work / "lst_ai.log").write_text(
        proc.stdout + "\n=== STDERR ===\n" + proc.stderr, encoding="utf-8", errors="replace"
    )
    elapsed = round(time.time() - start, 1)
    if proc.returncode != 0 or not annotated.is_file():
        return {"study": study, "status": "failed", "returncode": proc.returncode,
                "seconds": elapsed, "log": str(work / "lst_ai.log")}
    return {"study": study, "status": "ok", "seconds": elapsed,
            "annotated": str(annotated), "seg": str(out_dir / SEG)}


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    base = project_root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volumes", type=Path, default=base / "anatomy" / "volumes")
    parser.add_argument("--lst-root", type=Path, default=base / "lst_ai")
    parser.add_argument("--studies", nargs="*", default=None, help="Study names; default = all with T1.")
    parser.add_argument("--device", default="cpu", help='"cpu" or GPU id (e.g. "0").')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.studies:
        studies = args.studies
    else:
        studies = [
            p.name for p in sorted(args.volumes.iterdir())
            if (p / "NIFTI_T1").is_dir() and any((p / "NIFTI_T1").glob("*.nii*"))
        ]

    args.lst_root.mkdir(parents=True, exist_ok=True)
    for study in studies:
        result = run_one(
            study, args.volumes, args.lst_root,
            device=args.device, threshold=args.threshold, overwrite=args.overwrite,
        )
        print(result)


if __name__ == "__main__":
    main()
