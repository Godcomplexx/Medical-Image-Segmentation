"""
Convert a FastSurfer segmentation into a FLAIR-aligned aseg volume.

FastSurfer's ``--seg_only`` output ``aparc.DKTatlas+aseg.deep.mgz`` lives in
FastSurfer's conformed T1 space and encodes the cortex with DKT parcel IDs
(left 1000-1035, right 2000-2035) instead of the plain aseg cortex labels
3/42 that :mod:`synthseg_anatomy` expects. This module

1. remaps the DKT cortical parcels back to aseg 3 (left) / 42 (right), leaving
   ventricles / cerebellum / brainstem on their standard aseg IDs (FastSurfer
   keeps those unchanged), and
2. resamples the result onto the reconstructed FLAIR grid (nearest-neighbour,
   header affines) so ``seg[:, :, z]`` lines up with ``slice_{z}`` exactly like
   the SynthSeg path.

The output is a drop-in replacement for a SynthSeg segmentation: the same
``lesion_features_from_seg`` / ``classify_by_contact`` code and the same
``--synthseg-dir`` evaluation flag work unchanged, now backed by sharper
T1-derived cortex and ventricle boundaries (the point of the FastSurfer track:
juxtacortical ``U`` vs subcortical ``S``).

T1 and FLAIR come from the same scanning session, so the shared scanner world
coordinates make header-based resampling a reasonable T1->FLAIR alignment
(the same assumption ``resample_from_to`` relies on); it does not correct for
inter-sequence head motion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .synthseg_anatomy import resample_segmentation_to_reference


def remap_dkt_to_aseg(seg: np.ndarray) -> np.ndarray:
    """DKT cortical parcels -> aseg cortex (3 left / 42 right); other IDs kept."""
    out = seg.copy()
    out[(seg >= 1000) & (seg < 2000)] = 3
    out[(seg >= 2000) & (seg < 3000)] = 42
    return out


def fastsurfer_seg_to_flair(
    mgz_path: Path,
    flair_path: Path,
    output_path: Path,
) -> Path:
    """
    Remap DKT cortex and resample a FastSurfer seg onto the FLAIR grid.

    ``mgz_path`` is ``.../{sid}/mri/aparc.DKTatlas+aseg.deep.mgz``; ``flair_path``
    is the reconstructed FLAIR used to generate the slice PNGs. Writes a
    ``.nii.gz`` aligned to the FLAIR grid and returns its path.
    """
    import nibabel as nib

    seg_img = nib.load(str(mgz_path))
    remapped = remap_dkt_to_aseg(np.asarray(seg_img.get_fdata()).astype(np.int32))
    remapped_img = nib.Nifti1Image(remapped, seg_img.affine, seg_img.header)

    tmp = output_path.with_suffix(".remap.nii.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(remapped_img, str(tmp))
    try:
        resample_segmentation_to_reference(tmp, flair_path, output_path)
    finally:
        tmp.unlink(missing_ok=True)
    return output_path


__all__ = ["remap_dkt_to_aseg", "fastsurfer_seg_to_flair"]
