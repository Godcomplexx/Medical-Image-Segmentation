"""
Anatomy-aware features for I/P/S/U lesion classification.

The lesion classes are defined by anatomical position, not local appearance:

- ``P`` periventricular  — lesion abuts a ventricle;
- ``U`` juxtacortical    — lesion touches the cortex (brain boundary);
- ``I`` infratentorial   — brainstem / cerebellum, driven by slice level;
- ``S`` subcortical      — white matter, neither P nor U.

A tight crop around the lesion cannot express these relations, which is why an
appearance-only classifier stalls. This module derives geometric context from
the *full* slice plus the slice z-level so the class rule has something to stand
on.

The features are deliberately polarity-robust: the pipeline reconstructs T2/FLAIR
volumes (``create_t2_nifti``), and CSF is bright on T2 but dark on FLAIR, so we do
not hard-assume ventricle intensity. The strong, robust signals are:

- ``z_norm``                — slice position in the study (infratentorial → low/high extreme);
- ``dist_cortex_norm``      — distance from lesion to the brain boundary (juxtacortical → ~0);
- ``radial_norm``           — distance from lesion to brain centroid / brain radius
                              (periventricular lesions sit centrally, near the ventricles);
- ``dist_ventricle_norm``   — distance to a best-effort central-CSF proxy (optional, may be NaN).

This same feature contract can later be fed exact FastSurfer/SynthSeg structure
masks (ventricles, cortex, cerebellum, brainstem) without changing downstream code.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from .data import annotation_box, parse_yolo_segmentation


# Ordered feature contract shared by training and inference.
FEATURE_NAMES: tuple[str, ...] = (
    "z_norm",
    "dist_cortex_norm",
    "radial_norm",
    "dist_ventricle_norm",
    "lesion_area_norm",
    "centroid_x_norm",
    "centroid_y_norm",
    # Box-boundary contact: minimum distance from the lesion *box region* (not just
    # its centroid) to the cortex / ventricle. A juxtacortical (U) lesion touches
    # the cortex at its edge even when its centroid sits a few px inside white
    # matter — the centroid-sampled ``dist_cortex_norm`` misses that, which is why
    # U was the weak class. These edge-contact features separate juxtacortical
    # (min≈0) from subcortical (min>0), mirroring LST-AI's dilated-contact rule.
    "cortex_contact_norm",
    "ventricle_contact_norm",
    # Infratentorial contact (doctor's ГЛАВНАЯ ПРАВКА for class I): minimum distance
    # from the lesion box to the cerebellum/brainstem mask, normalized. ~0 when the
    # lesion sits in the infratentorial compartment → I, defined by real anatomy
    # instead of the z_norm proxy (a low temporal/occipital lobe lesion is also on a
    # low slice but is supratentorial). NaN when no structure mask is available
    # (FLAIR-proxy path), so the model's imputation keeps working.
    "infra_contact_norm",
)

_SLICE_RE = re.compile(r"^slice_(\d+)_(.+)$")


@dataclass(frozen=True)
class LesionFeatures:
    z_norm: float
    dist_cortex_norm: float
    radial_norm: float
    dist_ventricle_norm: float
    lesion_area_norm: float
    centroid_x_norm: float
    centroid_y_norm: float
    cortex_contact_norm: float = float("nan")
    ventricle_contact_norm: float = float("nan")
    infra_contact_norm: float = float("nan")

    def vector(self) -> list[float]:
        return [getattr(self, name) for name in FEATURE_NAMES]


def parse_study_slice(name: str) -> tuple[str, int]:
    """Return ``(study_id, slice_index)`` from a ``slice_{z}_{study}`` stem/path."""
    stem = Path(name).stem
    match = _SLICE_RE.match(stem)
    if not match:
        raise ValueError(f"Unrecognized slice filename: {name!r}")
    return match.group(2), int(match.group(1))


def compute_z_extent(names: Iterable[str]) -> dict[str, tuple[int, int]]:
    """Map each study to the ``(min, max)`` slice index observed among ``names``."""
    extent: dict[str, tuple[int, int]] = {}
    for name in names:
        try:
            study, z = parse_study_slice(name)
        except ValueError:
            continue
        low, high = extent.get(study, (z, z))
        extent[study] = (min(low, z), max(high, z))
    return extent


def brain_mask(gray: np.ndarray) -> np.ndarray:
    """Largest filled foreground component — the intracranial region."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu separates the (dark) background from head tissue robustly for either
    # T2 or FLAIR because it only needs foreground-vs-air contrast.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(gray, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    # Fill interior holes so ventricles/CSF do not punch through the brain body.
    filled = mask.copy()
    flood = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 1)
    holes = (filled == 0).astype(np.uint8)
    return ((mask | holes) > 0).astype(np.uint8)


def ventricle_proxy(gray: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """
    Best-effort central-CSF (ventricle) mask, polarity-agnostic.

    Tries both a dark-CSF (FLAIR) and bright-CSF (T2) hypothesis inside the brain
    interior and keeps whichever extreme forms the more central, compact blob.
    Returns an empty mask when neither hypothesis is convincing.
    """
    interior = cv2.erode(brain, np.ones((9, 9), np.uint8), iterations=1)
    if interior.sum() < 50:
        return np.zeros_like(brain)
    values = gray[interior > 0]
    ys, xs = np.nonzero(brain)
    cy, cx = float(ys.mean()), float(xs.mean())

    def _score(candidate: np.ndarray) -> float:
        area = float(candidate.sum())
        if area < 20:
            return -1.0
        cys, cxs = np.nonzero(candidate)
        radius = math.hypot(cys.mean() - cy, cxs.mean() - cx)
        brain_radius = math.sqrt(brain.sum() / math.pi) + 1e-6
        # Reward central, moderately sized blobs.
        centrality = 1.0 - min(radius / brain_radius, 1.0)
        size_ok = min(area / (0.02 * brain.sum() + 1e-6), 1.0)
        return centrality * size_ok

    dark = (gray <= np.percentile(values, 12)) & (interior > 0)
    bright = (gray >= np.percentile(values, 88)) & (interior > 0)
    best_mask = np.zeros_like(brain)
    best_score = 0.05  # require a minimal convincing score
    for candidate in (dark.astype(np.uint8), bright.astype(np.uint8)):
        opened = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        score = _score(opened)
        if score > best_score:
            best_score, best_mask = score, opened
    return best_mask


def _distance_to_mask(point: tuple[float, float], mask: np.ndarray) -> float | None:
    if mask is None or mask.sum() == 0:
        return None
    dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    x = int(min(max(round(point[0]), 0), mask.shape[1] - 1))
    y = int(min(max(round(point[1]), 0), mask.shape[0] - 1))
    return float(dist[y, x])


@dataclass
class SliceContext:
    """Per-slice anatomy derived once and reused across all lesions on the slice."""

    height: int
    width: int
    brain_cx: float
    brain_cy: float
    brain_radius: float
    edge_distance: np.ndarray | None
    ventricle_distance: np.ndarray | None
    # Distance transform to the infratentorial (cerebellum + brainstem) mask; only
    # available from real structure masks (SynthSeg). None on the FLAIR-proxy path.
    infra_distance: np.ndarray | None = None


def build_slice_context(gray: np.ndarray) -> SliceContext:
    """Segment the brain / ventricles once and precompute distance transforms."""
    height, width = gray.shape[:2]
    brain = brain_mask(gray)
    if brain.sum() == 0:
        return SliceContext(
            height=height,
            width=width,
            brain_cx=width / 2.0,
            brain_cy=height / 2.0,
            brain_radius=math.hypot(width, height) / 2.0,
            edge_distance=None,
            ventricle_distance=None,
        )
    ys, xs = np.nonzero(brain)
    edge = brain - cv2.erode(brain, np.ones((3, 3), np.uint8), iterations=1)
    edge_distance = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 5)
    ventricles = ventricle_proxy(gray, brain)
    ventricle_distance = (
        cv2.distanceTransform((ventricles == 0).astype(np.uint8), cv2.DIST_L2, 5)
        if ventricles.sum() > 0
        else None
    )
    return SliceContext(
        height=height,
        width=width,
        brain_cx=float(xs.mean()),
        brain_cy=float(ys.mean()),
        brain_radius=math.sqrt(brain.sum() / math.pi) + 1e-6,
        edge_distance=edge_distance,
        ventricle_distance=ventricle_distance,
    )


def slice_context_from_masks(
    brain: np.ndarray,
    cortex: np.ndarray,
    ventricles: np.ndarray,
    infratentorial: np.ndarray | None = None,
) -> SliceContext:
    """
    Build a :class:`SliceContext` from real structure masks (e.g. SynthSeg).

    Drop-in replacement for :func:`build_slice_context`: ``edge_distance`` becomes
    the true distance to the cortical ribbon (juxtacortical / U) and
    ``ventricle_distance`` the true distance to the ventricles (periventricular /
    P), so :func:`features_from_context` yields the same feature contract with
    anatomically exact inputs. Empty masks fall back to ``None`` (feature NaN /
    zero), keeping the model's imputation path intact.
    """
    brain = (brain > 0).astype(np.uint8)
    height, width = brain.shape[:2]
    if brain.sum() == 0:
        return SliceContext(
            height=height,
            width=width,
            brain_cx=width / 2.0,
            brain_cy=height / 2.0,
            brain_radius=math.hypot(width, height) / 2.0,
            edge_distance=None,
            ventricle_distance=None,
        )
    ys, xs = np.nonzero(brain)
    cortex = (cortex > 0).astype(np.uint8)
    ventricles = (ventricles > 0).astype(np.uint8)
    cortex_distance = (
        cv2.distanceTransform((cortex == 0).astype(np.uint8), cv2.DIST_L2, 5)
        if cortex.sum() > 0
        else None
    )
    ventricle_distance = (
        cv2.distanceTransform((ventricles == 0).astype(np.uint8), cv2.DIST_L2, 5)
        if ventricles.sum() > 0
        else None
    )
    infra_distance = None
    if infratentorial is not None:
        infra = (infratentorial > 0).astype(np.uint8)
        if infra.sum() > 0:
            infra_distance = cv2.distanceTransform((infra == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return SliceContext(
        height=height,
        width=width,
        brain_cx=float(xs.mean()),
        brain_cy=float(ys.mean()),
        brain_radius=math.sqrt(brain.sum() / math.pi) + 1e-6,
        edge_distance=cortex_distance,
        ventricle_distance=ventricle_distance,
        infra_distance=infra_distance,
    )


def _sample(distance: np.ndarray | None, cx: float, cy: float) -> float | None:
    if distance is None:
        return None
    x = int(min(max(round(cx), 0), distance.shape[1] - 1))
    y = int(min(max(round(cy), 0), distance.shape[0] - 1))
    return float(distance[y, x])


def _sample_box_min(
    distance: np.ndarray | None, box: Sequence[float]
) -> float | None:
    """Minimum distance over the lesion box region (edge contact, not centroid)."""
    if distance is None:
        return None
    h, w = distance.shape[:2]
    x1, y1, x2, y2 = box
    xa = int(min(max(math.floor(x1), 0), w - 1))
    xb = int(min(max(math.ceil(x2), 0), w - 1))
    ya = int(min(max(math.floor(y1), 0), h - 1))
    yb = int(min(max(math.ceil(y2), 0), h - 1))
    region = distance[ya : yb + 1, xa : xb + 1]
    if region.size == 0:
        return _sample(distance, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return float(region.min())


def features_from_context(
    context: SliceContext,
    lesion_box: Sequence[float],
    *,
    z_norm: float,
) -> LesionFeatures:
    """Compute the feature vector for one lesion given a precomputed context."""
    x1, y1, x2, y2 = lesion_box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    radial = math.hypot(cx - context.brain_cx, cy - context.brain_cy) / context.brain_radius
    boundary_dist = _sample(context.edge_distance, cx, cy)
    vent_dist = _sample(context.ventricle_distance, cx, cy)
    vent_norm = (vent_dist / context.brain_radius) if vent_dist is not None else float("nan")
    cortex_contact = _sample_box_min(context.edge_distance, lesion_box)
    vent_contact = _sample_box_min(context.ventricle_distance, lesion_box)
    infra_contact = _sample_box_min(context.infra_distance, lesion_box)
    lesion_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return LesionFeatures(
        z_norm=float(np.clip(z_norm, 0.0, 1.0)),
        dist_cortex_norm=float(min((boundary_dist or 0.0) / context.brain_radius, 2.0)),
        radial_norm=float(min(radial, 2.0)),
        dist_ventricle_norm=float(vent_norm),
        lesion_area_norm=float(lesion_area / (context.width * context.height)),
        centroid_x_norm=float(cx / context.width),
        centroid_y_norm=float(cy / context.height),
        cortex_contact_norm=(
            float(min(cortex_contact / context.brain_radius, 2.0))
            if cortex_contact is not None
            else float("nan")
        ),
        ventricle_contact_norm=(
            float(min(vent_contact / context.brain_radius, 2.0))
            if vent_contact is not None
            else float("nan")
        ),
        infra_contact_norm=(
            float(min(infra_contact / context.brain_radius, 2.0))
            if infra_contact is not None
            else float("nan")
        ),
    )


def lesion_features(
    gray: np.ndarray,
    lesion_box: Sequence[float],
    *,
    z_norm: float,
) -> LesionFeatures:
    """Compute the anatomy feature vector for one lesion box on a full slice."""
    return features_from_context(build_slice_context(gray), lesion_box, z_norm=z_norm)


def features_for_label_object(
    source_image: Path,
    source_label: Path,
    object_index: int,
    *,
    z_norm: float,
) -> LesionFeatures:
    """Load a slice + label and compute features for one annotated lesion."""
    annotations = parse_yolo_segmentation(source_label)
    annotation = annotations[object_index]
    with Image.open(source_image) as image:
        gray = np.asarray(image.convert("L"))
        box = annotation_box(annotation, image.width, image.height)
    return lesion_features(gray, box, z_norm=z_norm)


def classify_by_rule(
    features: LesionFeatures,
    *,
    infratentorial_z: float = 0.28,
    cortex_touch: float = 0.06,
    ventricle_touch: float = 0.10,
    periventricular_radial: float = 0.42,
) -> str:
    """
    Deterministic clinical fallback with priority I -> U -> P -> S.

    Thresholds are on normalized features and should be tuned on validation only.
    ``dist_ventricle_norm`` may be NaN when the CSF proxy fails; in that case P is
    decided from the radial position alone.
    """
    if features.z_norm <= infratentorial_z:
        return "I"
    if features.dist_cortex_norm <= cortex_touch:
        return "U"
    vent = features.dist_ventricle_norm
    near_ventricle = (not math.isnan(vent) and vent <= ventricle_touch) or (
        math.isnan(vent) and features.radial_norm <= periventricular_radial
    )
    if near_ventricle:
        return "P"
    return "S"


__all__ = [
    "FEATURE_NAMES",
    "LesionFeatures",
    "SliceContext",
    "parse_study_slice",
    "compute_z_extent",
    "brain_mask",
    "ventricle_proxy",
    "build_slice_context",
    "slice_context_from_masks",
    "features_from_context",
    "lesion_features",
    "features_for_label_object",
    "classify_by_rule",
]
