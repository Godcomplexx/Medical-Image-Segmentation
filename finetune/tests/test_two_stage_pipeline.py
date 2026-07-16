from pathlib import Path

import pytest
from PIL import Image

from finetune.two_stage_pipeline.data import (
    box_iou,
    contextual_square_box,
    prepare_two_stage_data,
)
import numpy as np

from finetune.two_stage_pipeline.anatomy import (
    build_slice_context,
    classify_by_rule,
    compute_z_extent,
    features_from_context,
    parse_study_slice,
)
from finetune.two_stage_pipeline.evaluate import _match_image


def _disc_brain_slice(size: int = 200, radius: int = 70) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    disc = (grid_x - size // 2) ** 2 + (grid_y - size // 2) ** 2 <= radius**2
    image = np.zeros((size, size), dtype=np.uint8)
    image[disc] = 150
    return image


def test_parse_study_slice_and_z_extent() -> None:
    assert parse_study_slice("slice_29_CAMMS03409_6013_Ext-mET.png") == (
        "CAMMS03409_6013_Ext-mET",
        29,
    )
    assert compute_z_extent(["slice_0_A", "slice_9_A", "slice_3_B"]) == {
        "A": (0, 9),
        "B": (3, 3),
    }


def test_anatomy_features_and_rule_encode_class_geometry() -> None:
    context = build_slice_context(_disc_brain_slice())
    assert abs(context.brain_cx - 100) < 2 and abs(context.brain_cy - 100) < 2

    # A lesion touching the brain boundary is juxtacortical (U).
    edge = features_from_context(context, (97, 29, 103, 35), z_norm=0.6)
    assert edge.dist_cortex_norm < 0.1
    assert classify_by_rule(edge) == "U"

    # A central lesion sits near the ventricles (P).
    center = features_from_context(context, (97, 97, 103, 103), z_norm=0.6)
    assert center.radial_norm < 0.1
    assert classify_by_rule(center) == "P"

    # A low slice level is infratentorial (I) regardless of in-plane position.
    assert classify_by_rule(features_from_context(context, (97, 97, 103, 103), z_norm=0.05)) == "I"


def _make_split(root: Path, split: str) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    positive_stem = f"slice_1_{split}_case"
    background_stem = f"slice_2_{split}_case"
    Image.new("L", (64, 64), color=80).save(image_dir / f"{positive_stem}.png")
    Image.new("L", (64, 64), color=20).save(image_dir / f"{background_stem}.png")
    (label_dir / f"{positive_stem}.txt").write_text(
        "1 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60\n"
        "4 0.70 0.70 0.80 0.70 0.80 0.80 0.70 0.80\n",
        encoding="utf-8",
    )
    (label_dir / f"{background_stem}.txt").write_text("", encoding="utf-8")


def test_contextual_square_box_stays_inside_image() -> None:
    assert contextual_square_box((0, 0, 4, 8), 64, 64, context=2, minimum_side=16) == (
        0,
        0,
        16,
        16,
    )
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175


def test_prepare_two_stage_data_preserves_split_and_builds_both_tasks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _make_split(source, "finetune")
    _make_split(source, "val")
    output = tmp_path / "output"

    summary = prepare_two_stage_data(
        source,
        output,
        minimum_side=16,
        train_background_ratio=0.5,
        val_background_ratio=0.5,
    )

    detector_label = (
        output / "detector" / "labels" / "train" / "slice_1_finetune_case.txt"
    )
    assert {
        line.split()[0]
        for line in detector_label.read_text(encoding="utf-8").splitlines()
    } == {"0"}
    assert (output / "detector" / "dataset.yaml").is_file()
    assert len(list((output / "classifier" / "images" / "train" / "I").glob("*.png"))) == 1
    assert len(list((output / "classifier" / "images" / "train" / "U").glob("*.png"))) == 1
    assert len(
        list((output / "classifier" / "images" / "train" / "background").glob("*.png"))
    ) == 1
    assert summary["splits"]["train"]["positive_crops"] == 2
    assert summary["splits"]["val"]["positive_crops"] == 2


def test_prepare_allows_preexisting_ultralytics_settings_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _make_split(source, "finetune")
    _make_split(source, "val")
    output = tmp_path / "output"
    settings = output / "ultralytics"
    settings.mkdir(parents=True)
    (settings / "settings.json").write_text("{}", encoding="utf-8")

    prepare_two_stage_data(
        source,
        output,
        minimum_side=16,
        train_background_ratio=0,
        val_background_ratio=0,
    )

    assert (output / "detector" / "dataset.yaml").is_file()


def _make_excluded_balance(root: Path) -> None:
    """A P-class slice living in excluded_balance/{images,labels}."""
    image_dir = root / "excluded_balance" / "images"
    label_dir = root / "excluded_balance" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    stem = "slice_5_excluded_case"
    Image.new("L", (64, 64), color=90).save(image_dir / f"{stem}.png")
    (label_dir / f"{stem}.txt").write_text(
        "2 0.30 0.30 0.50 0.30 0.50 0.50 0.30 0.50\n",
        encoding="utf-8",
    )


def _add_repeat_copy(root: Path) -> None:
    """A __repeat_ duplicate of the finetune I-class slice."""
    image_dir = root / "images" / "finetune"
    label_dir = root / "labels" / "finetune"
    stem = "slice_1_finetune_case__repeat_1"
    Image.new("L", (64, 64), color=80).save(image_dir / f"{stem}.png")
    (label_dir / f"{stem}.txt").write_text(
        "1 0.40 0.40 0.60 0.40 0.60 0.60 0.40 0.60\n",
        encoding="utf-8",
    )


def test_full_build_merges_excluded_balance_and_drops_repeats(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_split(source, "finetune")
    _make_split(source, "val")
    _make_excluded_balance(source)
    _add_repeat_copy(source)
    output = tmp_path / "output"

    summary = prepare_two_stage_data(
        source,
        output,
        minimum_side=16,
        full=True,
        train_background_ratio=0,
        val_background_ratio=0,
    )

    assert summary["full"] is True
    assert summary["exclude_repeats"] is True
    # 2 finetune slices + 1 excluded_balance slice; the __repeat_ copy is dropped.
    assert summary["splits"]["train"]["source_images"] == 3
    # The excluded_balance P lesion reaches the train classifier.
    assert len(list((output / "classifier" / "images" / "train" / "P").glob("*.png"))) == 1
    # Only the original I lesion, not its __repeat_ duplicate.
    assert len(list((output / "classifier" / "images" / "train" / "I").glob("*.png"))) == 1


def test_legacy_build_keeps_repeats_and_ignores_excluded_balance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_split(source, "finetune")
    _make_split(source, "val")
    _make_excluded_balance(source)
    _add_repeat_copy(source)
    output = tmp_path / "output"

    summary = prepare_two_stage_data(
        source,
        output,
        minimum_side=16,
        train_background_ratio=0,
        val_background_ratio=0,
    )

    assert summary["full"] is False
    assert summary["exclude_repeats"] is False
    # The __repeat_ copy is retained and excluded_balance is not merged.
    assert len(list((output / "classifier" / "images" / "train" / "I").glob("*.png"))) == 2
    assert not (output / "classifier" / "images" / "train" / "P").exists()


def test_full_build_rejects_train_val_leakage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_split(source, "finetune")
    _make_split(source, "val")
    _make_excluded_balance(source)
    # A validation slice reusing an excluded_balance basename would leak.
    leak_stem = "slice_5_excluded_case"
    Image.new("L", (64, 64), color=10).save(
        source / "images" / "val" / f"{leak_stem}.png"
    )
    (source / "labels" / "val" / f"{leak_stem}.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="leakage"):
        prepare_two_stage_data(
            source,
            tmp_path / "output",
            minimum_side=16,
            full=True,
            train_background_ratio=0,
            val_background_ratio=0,
        )


def test_full_build_rejects_duplicate_basename_across_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_split(source, "finetune")
    _make_split(source, "val")
    _make_excluded_balance(source)
    # excluded_balance reusing a finetune basename is an unresolvable ambiguity.
    dup_stem = "slice_1_finetune_case"
    Image.new("L", (64, 64), color=90).save(
        source / "excluded_balance" / "images" / f"{dup_stem}.png"
    )
    (source / "excluded_balance" / "labels" / f"{dup_stem}.txt").write_text(
        "2 0.30 0.30 0.50 0.30 0.50 0.50 0.30 0.50\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate basename"):
        prepare_two_stage_data(
            source,
            tmp_path / "output",
            minimum_side=16,
            full=True,
            train_background_ratio=0,
            val_background_ratio=0,
        )


def test_synthseg_masks_drive_class_geometry() -> None:
    from finetune.two_stage_pipeline.anatomy import classify_by_rule
    from finetune.two_stage_pipeline.synthseg_anatomy import (
        is_infratentorial,
        lesion_features_from_seg,
        structure_masks,
    )

    # One axial plane: cortex ribbon (id 42) rings a ventricle block (id 43) in WM.
    plane = np.zeros((100, 100), dtype=np.int32)
    plane[20:80, 20:80] = 2  # cerebral white matter (brain, not cortex/ventricle)
    plane[20:24, 20:80] = 42  # cortical ribbon along the top edge
    plane[45:55, 45:55] = 43  # lateral ventricle block in the centre
    seg = np.stack([plane] * 5, axis=2)  # depth 5

    masks = structure_masks(plane)
    assert masks["cortex"].sum() > 0 and masks["ventricles"].sum() > 0

    # Lesion on the cortical ribbon -> juxtacortical U.
    cortex_feat = lesion_features_from_seg(seg, 2, (48, 20, 52, 24), z_norm=0.6)
    assert cortex_feat.dist_cortex_norm < 0.1
    assert classify_by_rule(cortex_feat) == "U"

    # Lesion on the ventricle block -> periventricular P.
    vent_feat = lesion_features_from_seg(seg, 2, (48, 48, 52, 52), z_norm=0.6)
    assert vent_feat.dist_ventricle_norm < 0.1
    assert classify_by_rule(vent_feat) == "P"

    # Infratentorial membership from a cerebellum label (id 47).
    infra_plane = np.zeros((100, 100), dtype=np.int32)
    infra_plane[40:60, 40:60] = 47
    assert is_infratentorial(infra_plane, (48, 48, 52, 52)) is True
    assert is_infratentorial(plane, (48, 48, 52, 52)) is False


def test_match_image_requires_correct_class_and_iou() -> None:
    predictions = [
        {"class": "I", "joint_score": 0.9, "box_xyxy": [10, 10, 20, 20]},
        {"class": "P", "joint_score": 0.8, "box_xyxy": [30, 30, 40, 40]},
        {"class": "U", "joint_score": 0.7, "box_xyxy": [0, 0, 4, 4]},
    ]
    truth = [
        ("I", (10, 10, 20, 20)),
        ("S", (30, 30, 40, 40)),
    ]

    tp, fp, fn = _match_image(predictions, truth, iou_threshold=0.5)

    assert tp["I"] == 1
    assert fp["P"] == 1
    assert fp["U"] == 1
    assert fn["S"] == 1
