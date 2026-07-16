"""
Reconstruct a T1-weighted NIfTI per study for the LST-AI pipeline.

LST-AI needs a T1 alongside the FLAIR (already reconstructed under
``anatomy/volumes/{study}/NIFTI/*_flair.nii.gz``). The historical ``t1_filter``
required an *axial* series with ``0.8 < SliceThickness < 1.2`` and threw
``AttributeError`` when those tags were absent — which is the case for these
Philips studies (SliceThickness/EchoTime come back ``None``), so it matched zero
series even though a 3D T1 is present in every study.

This module picks the **single best T1 series** by description instead:

* require ``t1`` in the SeriesDescription;
* reject post-contrast / GAD / SPIR fat-sat series (they hurt registration and
  the McDonald zone annotation);
* prefer a 3D isotropic series (``3d`` in description), then the one with the
  most slices.

The chosen series is converted with the same deterministic ``dicom2nifti``
(reorient=True) path used for the FLAIR, so both volumes land in the same
canonical (RAS) orientation LST-AI expects. LST-AI still runs its own T1->FLAIR
registration, so exact grid agreement is not required.

    python -m finetune.two_stage_pipeline.lst_ai.reconstruct_t1 --studies +9 +219
"""
from __future__ import annotations

import argparse
import collections
import shutil
import tempfile
from pathlib import Path

import dicom2nifti
import pydicom

from finetune.src.filetools.dicom_to_nifti_converter import (
    _ascii_staging_root,
    configure_gdcmconv,
)

configure_gdcmconv()

_CONTRAST_MARKERS = ("gad", "post", "+c", "spir", "contrast", "+gad")


def _is_t1_candidate(desc: str) -> bool:
    desc = desc.lower()
    if "t1" not in desc:
        return False
    return not any(marker in desc for marker in _CONTRAST_MARKERS)


def choose_t1_series(study_dir: Path) -> tuple[str, dict] | None:
    """Return (SeriesInstanceUID, info) of the best plain-T1 series, or None."""
    series: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "desc": "", "files": []}
    )
    for path in study_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        desc = getattr(header, "SeriesDescription", "") or ""
        if not _is_t1_candidate(desc):
            continue
        uid = getattr(header, "SeriesInstanceUID", None)
        if uid is None:
            continue
        entry = series[uid]
        entry["n"] += 1
        entry["desc"] = desc
        entry["files"].append(path)
    if not series:
        return None

    def rank(item: tuple[str, dict]) -> tuple[int, int]:
        _, info = item
        is_3d = 1 if "3d" in info["desc"].lower() else 0
        return (is_3d, info["n"])  # prefer 3D, then most slices

    return max(series.items(), key=rank)


def reconstruct_t1(study_dir: Path, output_root: Path, *, overwrite: bool = False) -> Path | None:
    out_dir = output_root / study_dir.name / "NIFTI_T1"
    existing = sorted(out_dir.glob("*.nii*")) if out_dir.is_dir() else []
    if existing and not overwrite:
        return existing[0]

    chosen = choose_t1_series(study_dir)
    if chosen is None:
        return None
    uid, info = chosen

    staging_root = _ascii_staging_root()
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="t1_", dir=staging_root) as tmp:
        tmp_dir = Path(tmp)
        for i, src in enumerate(info["files"]):
            shutil.copy2(src, tmp_dir / f"{i:08d}.dcm")
        dicom2nifti.convert_directory(
            str(tmp_dir), str(out_dir), compression=True, reorient=True
        )

    produced = sorted(out_dir.glob("*.nii*"), key=lambda p: p.stat().st_size, reverse=True)
    if not produced:
        return None
    # Keep only the largest NIfTI (the chosen T1 volume); drop stray small ones.
    best = produced[0]
    print(f"  {study_dir.name}: T1 '{info['desc']}' ({info['n']} files) -> {best.name}")
    return best


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    default_input = project_root / "finetune" / "fileStorage" / "input_data"
    default_output = (
        project_root / "finetune" / "fileStorage" / "two_stage_full" / "anatomy" / "volumes"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--studies", nargs="*", default=None, help="Study folder names; default = all.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.studies:
        studies = [args.input / name for name in args.studies]
    else:
        studies = sorted(p for p in args.input.iterdir() if p.is_dir())

    ok = missing = failed = 0
    for study in studies:
        if not study.is_dir():
            print(f"  {study.name}: MISSING dir")
            missing += 1
            continue
        try:
            result = reconstruct_t1(study, args.output, overwrite=args.overwrite)
            if result is None:
                print(f"  {study.name}: no T1 series found")
                missing += 1
            else:
                ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  {study.name}: FAILED {str(exc)[:160]}")
            failed += 1
    print(f"done: ok={ok} missing={missing} failed={failed}")


if __name__ == "__main__":
    main()
