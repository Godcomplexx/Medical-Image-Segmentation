"""
Turn a SynthSeg brain segmentation into I/P/S/U anatomy features.

SynthSeg (contrast-agnostic, runs directly on FLAIR/T2 without T1) emits standard
FreeSurfer/aseg label IDs. This module groups those IDs into the structures the
lesion classes are defined against — lateral ventricles (P), cerebral cortex (U),
and the infratentorial compartment / cerebellum + brainstem (I) — and produces the
exact same :class:`~finetune.two_stage_pipeline.anatomy.LesionFeatures` contract as
the FLAIR-proxy path, so the trained classifier and cascade are unchanged.

## Producing an aligned segmentation

The slice PNGs are axial planes ``volume[:, :, z]`` of the reconstructed FLAIR/T2
volume (see ``finetune.src.yolo.image_generator.generate_images``). SynthSeg
conforms/resamples internally, so its output must be resampled back onto the
*original* volume grid (nearest-neighbour) before use — otherwise ``seg[:, :, z]``
will not line up with ``slice_{z}``. Use :func:`resample_segmentation_to_reference`
for that. Recommended run (in a separate env with SynthSeg installed, so the
training venv is untouched):

    mri_synthseg --i study_flair.nii --o study_seg.nii --robust --parc

then resample ``study_seg.nii`` to ``study_flair.nii`` and cache it per study.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .anatomy import LesionFeatures, features_from_context, slice_context_from_masks


# FreeSurfer / SynthSeg (aseg) label IDs.
LATERAL_VENTRICLE_IDS: frozenset[int] = frozenset({4, 43, 5, 44})  # + inf-lat-vent
MIDLINE_VENTRICLE_IDS: frozenset[int] = frozenset({14, 15})  # 3rd, 4th ventricle
VENTRICLE_IDS: frozenset[int] = LATERAL_VENTRICLE_IDS | MIDLINE_VENTRICLE_IDS
CEREBRAL_CORTEX_IDS: frozenset[int] = frozenset({3, 42})
CEREBELLUM_IDS: frozenset[int] = frozenset({7, 8, 46, 47})  # WM + cortex, both sides
BRAINSTEM_IDS: frozenset[int] = frozenset({16})
INFRATENTORIAL_IDS: frozenset[int] = CEREBELLUM_IDS | BRAINSTEM_IDS | frozenset({15})


def _mask_of(seg2d: np.ndarray, ids: frozenset[int]) -> np.ndarray:
    return np.isin(seg2d, list(ids)).astype(np.uint8)


def brain_z_extent(seg_volume: np.ndarray) -> tuple[int, int]:
    """First and last axial index containing brain, for consistent ``z_norm``.

    Using the brain span (rather than the raw volume depth) makes the
    infratentorial signal comparable across studies with different padding, and is
    computed identically at training and inference time.
    """
    present = np.any(seg_volume > 0, axis=(0, 1))
    indices = np.nonzero(present)[0]
    if indices.size == 0:
        return 0, seg_volume.shape[2] - 1
    return int(indices[0]), int(indices[-1])


def z_norm_from_extent(z: int, extent: tuple[int, int]) -> float:
    low, high = extent
    return 0.5 if high == low else float((z - low) / (high - low))


def structure_masks(seg2d: np.ndarray) -> dict[str, np.ndarray]:
    """Split one 2D label plane into brain / cortex / ventricle / infratentorial masks."""
    return {
        "brain": (seg2d > 0).astype(np.uint8),
        "cortex": _mask_of(seg2d, CEREBRAL_CORTEX_IDS),
        "ventricles": _mask_of(seg2d, VENTRICLE_IDS),
        "infratentorial": _mask_of(seg2d, INFRATENTORIAL_IDS),
    }


def is_infratentorial(
    seg2d: np.ndarray,
    lesion_box: tuple[float, float, float, float],
    *,
    dilation: int = 3,
) -> bool:
    """Whether a lesion centroid falls in the cerebellum / brainstem compartment."""
    import cv2

    infra = _mask_of(seg2d, INFRATENTORIAL_IDS)
    if infra.sum() == 0:
        return False
    if dilation > 0:
        infra = cv2.dilate(infra, np.ones((dilation, dilation), np.uint8))
    x1, y1, x2, y2 = lesion_box
    cx = int(min(max(round((x1 + x2) / 2.0), 0), infra.shape[1] - 1))
    cy = int(min(max(round((y1 + y2) / 2.0), 0), infra.shape[0] - 1))
    return bool(infra[cy, cx])


def classify_by_contact(
    seg2d: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    dilate_px: int = 3,
    priority: tuple[str, ...] = ("I", "P", "U"),
) -> str:
    """
    LST-AI / McDonald location rule: dilate the ventricle & cortex masks and test
    contact with the lesion.

    A lesion overlapping the cerebellum/brainstem is infratentorial (I); one
    touching the ~3 mm-dilated ventricle mask is periventricular (P); one touching
    the dilated cortex is juxtacortical (U); everything else is deep white matter
    (S). ``priority`` breaks ties for a lesion touching several regions. This is
    the crisp published rule, replacing soft distance thresholds — the contact
    test sharply separates juxtacortical (touches cortex) from subcortical (near
    but not touching).
    """
    import cv2

    lm = lesion_mask > 0
    if not lm.any():
        return "S"
    infra = _mask_of(seg2d, INFRATENTORIAL_IDS).astype(bool)
    ventricle = _mask_of(seg2d, VENTRICLE_IDS)
    cortex = _mask_of(seg2d, CEREBRAL_CORTEX_IDS)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        ventricle = cv2.dilate(ventricle, kernel)
        cortex = cv2.dilate(cortex, kernel)
    touches = {
        "I": bool((infra & lm).any()),
        "P": bool((ventricle.astype(bool) & lm).any()),
        "U": bool((cortex.astype(bool) & lm).any()),
    }
    for label in priority:
        if touches[label]:
            return label
    return "S"


def lesion_features_from_seg(
    seg_volume: np.ndarray,
    z: int,
    lesion_box: tuple[float, float, float, float],
    *,
    z_norm: float,
) -> LesionFeatures:
    """
    Anatomy features for one lesion from a segmentation volume aligned to the slices.

    ``seg_volume`` is the SynthSeg labels resampled onto the FLAIR/T2 grid used to
    generate the slice PNGs; ``z`` indexes the third axis exactly like ``slice_{z}``.
    """
    if not 0 <= z < seg_volume.shape[2]:
        raise IndexError(f"slice {z} outside segmentation depth {seg_volume.shape[2]}")
    seg2d = np.asarray(seg_volume[:, :, z])
    masks = structure_masks(seg2d)
    context = slice_context_from_masks(
        masks["brain"], masks["cortex"], masks["ventricles"], masks["infratentorial"]
    )
    return features_from_context(context, lesion_box, z_norm=z_norm)


def resample_segmentation_to_reference(
    segmentation_path: Path,
    reference_path: Path,
    output_path: Path,
) -> Path:
    """
    Nearest-neighbour resample a SynthSeg output onto the reference volume grid.

    Guarantees ``seg[:, :, z]`` aligns with ``slice_{z}`` of the reference FLAIR/T2.
    Requires ``nibabel`` and ``scipy`` (both already in the training env).
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    seg = nib.load(str(segmentation_path))
    reference = nib.load(str(reference_path))
    resampled = resample_from_to(seg, (reference.shape[:3], reference.affine), order=0)
    nib.save(resampled, str(output_path))
    return output_path


__all__ = [
    "VENTRICLE_IDS",
    "CEREBRAL_CORTEX_IDS",
    "INFRATENTORIAL_IDS",
    "structure_masks",
    "is_infratentorial",
    "classify_by_contact",
    "brain_z_extent",
    "z_norm_from_extent",
    "lesion_features_from_seg",
    "resample_segmentation_to_reference",
]
