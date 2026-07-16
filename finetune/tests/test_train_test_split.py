from pathlib import Path

from finetune.src.yolo.train_test_split import (
    balance_training_classes,
    balance_training_backgrounds,
    oversample_rare_classes,
    split_dataset,
)


def _case_id(path: Path) -> str:
    return path.stem.split("_", 2)[2]


def test_split_dataset_keeps_studies_and_pairs_together(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    for case_index in range(10):
        for slice_index in range(3):
            stem = f"slice_{slice_index}_case_{case_index:02d}"
            (images / f"{stem}.png").write_bytes(b"image")
            (labels / f"{stem}.txt").write_text("0 0.5 0.5\n", encoding="utf-8")

    split_dataset(tmp_path, train_ratio=0.8, val_ratio=0.2)

    train_images = list((images / "finetune").glob("*.png"))
    val_images = list((images / "val").glob("*.png"))
    train_cases = {_case_id(path) for path in train_images}
    val_cases = {_case_id(path) for path in val_images}

    assert len(train_images) == 24
    assert len(val_images) == 6
    assert train_cases.isdisjoint(val_cases)
    assert all((labels / "finetune" / f"{path.stem}.txt").exists() for path in train_images)
    assert all((labels / "val" / f"{path.stem}.txt").exists() for path in val_images)


def test_background_balancing_keeps_positive_and_neighbor_slices(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images" / "finetune"
    label_dir = tmp_path / "labels" / "finetune"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for slice_index in range(10):
        stem = f"slice_{slice_index}_case_a"
        (image_dir / f"{stem}.png").write_bytes(b"image")
        label = "1 0.1 0.1 0.2 0.1 0.2 0.2\n" if slice_index == 5 else ""
        (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")

    balance_training_backgrounds(
        tmp_path,
        background_to_positive_ratio=1.0,
        neighbor_radius=2,
    )

    remaining = {path.stem for path in label_dir.glob("*.txt")}
    assert "slice_5_case_a" in remaining
    assert {
        "slice_3_case_a",
        "slice_4_case_a",
        "slice_6_case_a",
        "slice_7_case_a",
    }.issubset(remaining)
    assert len(remaining) == 5


def test_rare_class_oversampling_duplicates_only_training_pairs(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images" / "finetune"
    label_dir = tmp_path / "labels" / "finetune"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for class_id in (0, 2, 3):
        stem = f"slice_{class_id}_case_a"
        (image_dir / f"{stem}.png").write_bytes(b"image")
        (label_dir / f"{stem}.txt").write_text(
            f"{class_id} 0.1 0.1 0.2 0.1 0.2 0.2\n",
            encoding="utf-8",
        )

    oversample_rare_classes(tmp_path, {0: 2, 3: 1})

    assert len(list(image_dir.glob("*.png"))) == 6
    assert len(list(label_dir.glob("*.txt"))) == 6
    assert not list(label_dir.glob("slice_2_case_a__repeat_*.txt"))


def test_class_balancing_keeps_rare_classes_and_uses_empty_backgrounds(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images" / "finetune"
    label_dir = tmp_path / "labels" / "finetune"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    class_ids = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    for slice_index, class_id in enumerate(class_ids):
        stem = f"slice_{slice_index}_case_a"
        (image_dir / f"{stem}.png").write_bytes(b"image")
        (label_dir / f"{stem}.txt").write_text(
            f"{class_id} 0.1 0.1 0.2 0.1 0.2 0.2\n",
            encoding="utf-8",
        )
    for slice_index in range(10, 16):
        stem = f"slice_{slice_index}_case_a"
        (image_dir / f"{stem}.png").write_bytes(b"image")
        (label_dir / f"{stem}.txt").write_text("", encoding="utf-8")

    summary = balance_training_classes(
        tmp_path,
        target_common_slices=2,
        background_to_positive_ratio=0.5,
        neighbor_radius=0,
    )

    retained = list(label_dir.glob("*.txt"))
    retained_ids = {
        int(line.split()[0])
        for label in retained
        for line in label.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert retained_ids == {0, 1, 2, 3}
    assert summary["class_1_slices"] == 2
    assert summary["class_2_slices"] == 2
    assert len(list(label_dir.glob("*__repeat_1.txt"))) == 2
    assert all(
        not label.read_text(encoding="utf-8").strip()
        for label in retained
        if label.stem.startswith(("slice_10_", "slice_11_", "slice_12_"))
    )
