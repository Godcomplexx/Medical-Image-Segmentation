"""
Own unsupervised MS-lesion detector (LST-LGA / SAMSEG core idea, pure Python).

Motivation: a detector *trained* on our annotation caps at the annotation's
completeness (our GT is sparse — see the unmarked-lesion montage). An
unsupervised, contrast-adaptive detector sidesteps that: it does not learn from
labels at all, and a single threshold ``kappa`` is calibrated on our data.

Algorithm (per study, in the reconstructed-FLAIR / SynthSeg grid):

1. Normal-appearing white matter (NAWM) statistics — robust median + MAD of FLAIR
   intensity inside the WM mask, iteratively excluding hyperintense outliers
   (candidate lesions) so the reference is truly *normal* WM. This makes the
   detector contrast-agnostic per study (no global intensity assumption), exactly
   like SAMSEG adapts to the sequence.
2. Eligible region — brain minus ventricles/CSF, restricted to WM dilated toward
   cortex and ventricles (so juxtacortical and periventricular lesions are kept)
   plus the infratentorial compartment.
3. Lesion belief — robust z-score ``(FLAIR - median) / (1.4826 * MAD)``; a voxel is
   a lesion seed when ``z > kappa`` inside the eligible region.
4. Growth + cleanup — seeds are grown into contiguous hyperintense voxels
   (``z > kappa_grow``), then morphological opening and a minimum 3-D size remove
   speckle. LGA's "grow along hyperintense FLAIR" step.

Output is a 3-D binary lesion volume per study, mirroring LST-AI's
``space-flair_seg-lst.nii.gz`` so the existing hybrid classifier/eval consume it
unchanged.

    python -m finetune.two_stage_pipeline.lst_ai.unsup_detect --kappa 2.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

# FreeSurfer/SynthSeg aseg labels.
CEREBRAL_WM = (2, 41)
CEREBELLAR_WM = (7, 46)
BRAINSTEM = (16,)
WM_HYPO = (77,)
WM_IDS = CEREBRAL_WM + CEREBELLAR_WM + BRAINSTEM + WM_HYPO
CORTEX_IDS = (3, 42, 8, 47)  # cerebral + cerebellar cortex
VENTRICLE_IDS = (4, 43, 5, 44, 14, 15)
INFRATENTORIAL_IDS = (7, 8, 46, 47, 16, 15)


def _robust_nawm(flair: np.ndarray, wm: np.ndarray, iters: int = 3) -> tuple[float, float]:
    """Median + MAD of FLAIR in WM, iteratively dropping hyperintense outliers."""
    vals = flair[wm > 0].astype(np.float64)
    if vals.size == 0:
        return float(np.median(flair)), float(np.std(flair) + 1e-6)
    for _ in range(iters):
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-6
        keep = vals < med + 3.0 * mad  # exclude candidate lesions from the reference
        if keep.sum() < 0.5 * vals.size:
            break
        vals = vals[keep]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-6
    return med, mad


def detect(
    flair: np.ndarray,
    seg: np.ndarray,
    *,
    kappa: float = 3.0,
    kappa_grow: float = 99.0,  # growth off by default; set < kappa to enable
    min_size: int = 5,
    dilate: int = 2,
) -> np.ndarray:
    """Return a 3-D binary lesion mask on the FLAIR/seg grid."""
    seg = np.asarray(seg)
    flair = np.asarray(flair).astype(np.float64)
    wm = np.isin(seg, WM_IDS)
    med, mad = _robust_nawm(flair, wm)
    z = (flair - med) / mad

    brain = seg > 0
    ventricles = np.isin(seg, VENTRICLE_IDS)
    # eligible = WM (dilated toward cortex/ventricles) + infratentorial, minus CSF
    eligible = wm | np.isin(seg, INFRATENTORIAL_IDS)
    if dilate > 0:
        eligible = ndimage.binary_dilation(eligible, iterations=dilate)
    eligible &= brain & ~ventricles

    lesion = (z > kappa) & eligible
    if kappa_grow < kappa:
        # LGA-style growth: extend each thresholded seed into contiguous, still
        # clearly-hyperintense voxels (z > kappa_grow), but only components that
        # already contain a seed — so kappa stays the controlling knob.
        grow = (z > kappa_grow) & eligible
        labels, n = ndimage.label(grow)
        if n:
            seed_labels = set(np.unique(labels[lesion])) - {0}
            lesion = np.isin(labels, list(seed_labels))
    # cleanup: opening + minimum 3-D size
    lesion = ndimage.binary_opening(lesion, iterations=1)
    labels, n = ndimage.label(lesion)
    if n:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
        small = {i + 1 for i, s in enumerate(sizes) if s < min_size}
        if small:
            lesion[np.isin(labels, list(small))] = False
    return lesion.astype(np.uint8)


def _load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata())


def run_all(volumes_root: Path, synthseg_dir: Path, out_dir: Path, *, kappa: float,
            studies: list[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if studies is None:
        studies = [p.stem[:-4] if p.name.endswith(".nii.gz") else p.stem
                   for p in synthseg_dir.glob("*.nii.gz")]
    ok = miss = 0
    for study in studies:
        seg_path = synthseg_dir / f"{study}.nii.gz"
        nifti_dir = volumes_root / study / "NIFTI"
        flairs = sorted(nifti_dir.glob("*flair*.nii*")) or sorted(nifti_dir.glob("*.nii*"))
        if not seg_path.is_file() or not flairs:
            miss += 1
            continue
        seg = _load(seg_path).astype(np.int32)
        flair = _load(flairs[0])
        if flair.shape != seg.shape:
            # SynthSeg is resampled to the FLAIR grid already; skip mismatches
            miss += 1
            continue
        mask = detect(flair, seg, kappa=kappa)
        ref = nib.load(str(seg_path))
        nib.save(nib.Nifti1Image(mask, ref.affine, ref.header), str(out_dir / f"{study}.nii.gz"))
        ok += 1
    return {"ok": ok, "missing": miss, "out": str(out_dir)}


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    base = root / "finetune" / "fileStorage" / "two_stage_full"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volumes", type=Path, default=base / "anatomy" / "volumes")
    parser.add_argument("--synthseg-dir", type=Path, default=base / "anatomy" / "segmentations")
    parser.add_argument("--out", type=Path, default=base / "unsup_lesion")
    parser.add_argument("--kappa", type=float, default=2.5)
    parser.add_argument("--studies", nargs="*", default=None)
    args = parser.parse_args()
    print(run_all(args.volumes, args.synthseg_dir, args.out, kappa=args.kappa, studies=args.studies))


if __name__ == "__main__":
    main()
