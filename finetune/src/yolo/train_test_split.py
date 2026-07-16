import math
import os
from pathlib import Path
import random
import re
import shutil
from sklearn.model_selection import train_test_split
from finetune.config.logs_config import *
from typing import List


def create_dirs(base_dir: Path, sub_dirs: List[str]) -> None:
    """
    Creates subdirectories inside a base directory.

    Args:
        base_dir (Path): The root directory where subdirectories will be created.
        sub_dirs (List[str]): List of subdirectory names to create.
    """
    for sub_dir in sub_dirs:
        try:
            dir_path = base_dir / sub_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            logger_processing_manager.debug(f"Created directory: {dir_path}")
        except Exception as e:
            logger_processing_manager.error(f"Failed to create directory: {str(e)}")


def move_and_rename_files(src_root: Path, dst_root: Path) -> None:
    """
    КОПИРУЕТ (не перемещает) и переименовывает image/label файлы из источника в приёмник.

    Копирование делает сборку `train_set` идемпотентной: `processed_data` остаётся
    источником, поэтому пересобирать датасет можно многократно без повторной
    обработки DICOM (раньше `move` уничтожал источник — при падении split/balance
    и повторном reset=True данные терялись).

    Имя включает имя каталога-исследования, чтобы избежать коллизий.
    """
    for file_path in src_root.rglob('*.*'):
        try:
            parent_name = file_path.parent.name
            if parent_name in ['images', 'labels']:
                new_name = f"{file_path.stem}_{file_path.parents[2].name}{file_path.suffix}"
                dst_path = dst_root / parent_name / new_name
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(file_path), str(dst_path))
                logger_processing_manager.debug(f"Copied and renamed {file_path} to {dst_path}")
        except Exception as e:
            logger_processing_manager.error(f"Failed to copy {file_path}: {str(e)}")


def split_dataset(base_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.2) -> None:
    """
    Splits dataset into finetune, validation, and test subsets.

    Args:
        base_dir (Path): Directory containing 'images' and 'labels' folders.
        train_ratio (float): Proportion of data to include in the training set.
        val_ratio (float): Proportion of data to include in the validation set.
    """
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1")
    if not math.isclose(train_ratio + val_ratio, 1.0, rel_tol=1e-5):
        raise ValueError("train_ratio + val_ratio must equal 1.0")

    image_extensions = {".png", ".jpg", ".jpeg"}
    image_files = [
        path
        for path in (base_dir / "images").rglob("*")
        if path.is_file() and path.suffix.casefold() in image_extensions
    ]
    label_files = [
        path
        for path in (base_dir / "labels").rglob("*.txt")
        if path.is_file()
    ]

    images_by_stem = _index_unique_files(image_files, "image")
    labels_by_stem = _index_unique_files(label_files, "label")
    missing_labels = sorted(images_by_stem.keys() - labels_by_stem.keys())
    missing_images = sorted(labels_by_stem.keys() - images_by_stem.keys())
    if missing_labels or missing_images:
        raise ValueError(
            "Image/label pairing failed: "
            f"{len(missing_labels)} images without labels, "
            f"{len(missing_images)} labels without images"
        )
    if not images_by_stem:
        raise ValueError(f"No paired YOLO files found in {base_dir}")

    case_by_stem = {stem: _case_id_from_stem(stem) for stem in images_by_stem}
    case_ids = sorted(set(case_by_stem.values()))
    train_cases, val_cases = train_test_split(
        case_ids,
        test_size=val_ratio,
        random_state=42,
    )
    split_by_case = {
        **{case_id: "finetune" for case_id in train_cases},
        **{case_id: "val" for case_id in val_cases},
    }

    for stem, image_path in images_by_stem.items():
        split_name = split_by_case[case_by_stem[stem]]
        _move_to_split(image_path, base_dir / "images" / split_name)
        _move_to_split(labels_by_stem[stem], base_dir / "labels" / split_name)

    # Ultralytics caches contain paths from the previous split.
    for cache_path in base_dir.rglob("*.cache"):
        cache_path.unlink()

    logger_processing_manager.info(
        "Study-level split completed: "
        f"{len(train_cases)} train cases, {len(val_cases)} validation cases, "
        f"{len(images_by_stem)} paired slices"
    )


def _index_unique_files(files: List[Path], category: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in files:
        if path.stem in indexed:
            raise ValueError(f"Duplicate {category} basename: {path.stem}")
        indexed[path.stem] = path
    return indexed


def _case_id_from_stem(stem: str) -> str:
    match = re.fullmatch(r"slice_\d+_(.+)", stem)
    if not match:
        raise ValueError(f"Cannot extract case ID from YOLO basename: {stem}")
    return match.group(1)


def _move_to_split(source: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        raise FileExistsError(f"Split destination already exists: {destination}")
    shutil.move(str(source), str(destination))


def balance_training_backgrounds(
    base_dir: Path,
    background_to_positive_ratio: float = 0.75,
    neighbor_radius: int = 2,
    seed: int = 42,
) -> None:
    """Limit distant empty slices while retaining hard negatives near lesions."""
    if background_to_positive_ratio <= 0:
        raise ValueError("background_to_positive_ratio must be positive")

    image_dir = base_dir / "images" / "finetune"
    label_dir = base_dir / "labels" / "finetune"
    records: list[tuple[Path, Path, str, int, bool]] = []
    positive_indices: dict[str, set[int]] = {}

    for label_path in sorted(label_dir.glob("*.txt")):
        match = re.fullmatch(r"slice_(\d+)_(.+)", label_path.stem)
        if not match:
            raise ValueError(f"Cannot parse training label name: {label_path.name}")
        slice_index = int(match.group(1))
        case_id = match.group(2)
        image_path = next(
            (
                image_dir / f"{label_path.stem}{extension}"
                for extension in (".png", ".jpg", ".jpeg")
                if (image_dir / f"{label_path.stem}{extension}").exists()
            ),
            None,
        )
        if image_path is None:
            raise FileNotFoundError(f"Missing image for {label_path.name}")
        is_positive = bool(label_path.read_text(encoding="utf-8").strip())
        records.append((image_path, label_path, case_id, slice_index, is_positive))
        if is_positive:
            positive_indices.setdefault(case_id, set()).add(slice_index)

    # Oversampled copies must not inflate the target number of backgrounds
    # when this function is rerun on an already prepared dataset.
    original_records = [
        record for record in records if "__repeat_" not in record[1].stem
    ]
    positives = [record for record in original_records if record[4]]
    backgrounds = [record for record in original_records if not record[4]]
    protected_backgrounds = [
        record
        for record in backgrounds
        if any(
            abs(record[3] - positive_index) <= neighbor_radius
            for positive_index in positive_indices.get(record[2], set())
        )
    ]
    protected_paths = {record[1] for record in protected_backgrounds}
    optional_backgrounds = [
        record for record in backgrounds if record[1] not in protected_paths
    ]

    target_backgrounds = max(
        len(protected_backgrounds),
        round(len(positives) * background_to_positive_ratio),
    )
    optional_to_keep = max(0, target_backgrounds - len(protected_backgrounds))
    rng = random.Random(seed)
    kept_optional_paths = {
        record[1]
        for record in rng.sample(
            optional_backgrounds,
            min(optional_to_keep, len(optional_backgrounds)),
        )
    }

    removed = 0
    for image_path, label_path, _, _, _ in optional_backgrounds:
        if label_path in kept_optional_paths:
            continue
        image_path.unlink()
        label_path.unlink()
        removed += 1

    logger_processing_manager.info(
        "Background balancing completed: "
        f"{len(positives)} positive slices, "
        f"{len(backgrounds) - removed} retained background slices, "
        f"{removed} distant background slices removed"
    )


def oversample_rare_classes(
    base_dir: Path,
    repeats_by_class: dict[int, int] | None = None,
) -> None:
    """Duplicate rare-class training pairs before online augmentation."""
    # nc=5 схема: редкие медицинские классы I и U — это 1 и 4.
    repeats_by_class = repeats_by_class or {1: 2, 4: 1}
    image_dir = base_dir / "images" / "finetune"
    label_dir = base_dir / "labels" / "finetune"
    created = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        if "__repeat_" in label_path.stem:
            continue
        lines = label_path.read_text(encoding="utf-8").splitlines()
        classes = {int(line.split()[0]) for line in lines if line.strip()}
        repeat_count = max(
            (repeats_by_class.get(class_id, 0) for class_id in classes),
            default=0,
        )
        if repeat_count == 0:
            continue

        image_path = next(
            (
                image_dir / f"{label_path.stem}{extension}"
                for extension in (".png", ".jpg", ".jpeg")
                if (image_dir / f"{label_path.stem}{extension}").exists()
            ),
            None,
        )
        if image_path is None:
            raise FileNotFoundError(f"Missing image for {label_path.name}")

        for repeat_index in range(1, repeat_count + 1):
            repeated_stem = f"{label_path.stem}__repeat_{repeat_index}"
            shutil.copy2(
                image_path,
                image_dir / f"{repeated_stem}{image_path.suffix}",
            )
            shutil.copy2(label_path, label_dir / f"{repeated_stem}.txt")
            created += 1

    for cache_path in (base_dir / "labels").glob("*.cache"):
        cache_path.unlink()

    logger_processing_manager.info(
        f"Rare-class oversampling created {created} additional training pairs"
    )


def balance_training_classes(
    base_dir: Path,
    target_common_slices: int = 800,
    background_to_positive_ratio: float = 0.5,
    neighbor_radius: int = 1,
    seed: int = 42,
) -> dict[str, int]:
    """
    Build a reversible, moderately balanced training subset.

    YOLO IDs 0..3 represent I/P/S/U. Background is represented by an empty
    label, never by class 0. All slices containing rare classes I or U are
    retained. P/S-only slices and distant backgrounds are deterministically
    reduced, while one extra copy of each I slice is allowed.
    """
    if target_common_slices <= 0:
        raise ValueError("target_common_slices must be positive")
    if background_to_positive_ratio <= 0:
        raise ValueError("background_to_positive_ratio must be positive")

    image_dir = base_dir / "images" / "finetune"
    label_dir = base_dir / "labels" / "finetune"
    archive_root = base_dir / "excluded_balance"
    archive_image_dir = archive_root / "images"
    archive_label_dir = archive_root / "labels"

    # Restore previously excluded originals so rerunning is deterministic.
    for source_dir, destination_dir in (
        (archive_image_dir, image_dir),
        (archive_label_dir, label_dir),
    ):
        if not source_dir.exists():
            continue
        for source in source_dir.iterdir():
            destination = destination_dir / source.name
            if destination.exists():
                raise FileExistsError(f"Cannot restore balanced sample: {destination}")
            shutil.move(str(source), str(destination))

    # Repeats are derived data and are regenerated after balancing.
    for repeated_label in label_dir.glob("*__repeat_*.txt"):
        repeated_image = next(
            (
                image_dir / f"{repeated_label.stem}{extension}"
                for extension in (".png", ".jpg", ".jpeg")
                if (image_dir / f"{repeated_label.stem}{extension}").exists()
            ),
            None,
        )
        if repeated_image:
            repeated_image.unlink()
        repeated_label.unlink()

    records: list[tuple[Path, Path, str, int, frozenset[int]]] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        match = re.fullmatch(r"slice_(\d+)_(.+)", label_path.stem)
        if not match:
            raise ValueError(f"Cannot parse training label name: {label_path.name}")
        image_path = next(
            (
                image_dir / f"{label_path.stem}{extension}"
                for extension in (".png", ".jpg", ".jpeg")
                if (image_dir / f"{label_path.stem}{extension}").exists()
            ),
            None,
        )
        if image_path is None:
            raise FileNotFoundError(f"Missing image for {label_path.name}")
        classes = frozenset(
            int(line.split()[0])
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not classes <= {1, 2, 3, 4}:
            raise ValueError(
                f"Unexpected class IDs in {label_path.name}: {sorted(classes)}"
            )
        records.append(
            (image_path, label_path, match.group(2), int(match.group(1)), classes)
        )

    positives = [record for record in records if record[4]]
    backgrounds = [record for record in records if not record[4]]
    # nc=5: редкие классы I=1 и U=4; частые P=2 и S=3.
    rare_classes = {1, 4}
    common_classes = {2, 3}
    selected_positives = [
        record for record in positives if record[4] & rare_classes
    ]
    remaining = [
        record for record in positives if not record[4] & rare_classes
    ]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    presence = {
        class_id: sum(class_id in record[4] for record in selected_positives)
        for class_id in range(1, 5)
    }
    while remaining and any(
        presence[class_id] < target_common_slices
        for class_id in common_classes
    ):
        best_index = max(
            range(len(remaining)),
            key=lambda index: sum(
                class_id in remaining[index][4]
                and presence[class_id] < target_common_slices
                for class_id in common_classes
            ),
        )
        candidate = remaining[best_index]
        if not any(
            class_id in candidate[4]
            and presence[class_id] < target_common_slices
            for class_id in common_classes
        ):
            break
        selected_positives.append(remaining.pop(best_index))
        for class_id in candidate[4]:
            presence[class_id] += 1

    positive_indices: dict[str, set[int]] = {}
    for _, _, case_id, slice_index, _ in selected_positives:
        positive_indices.setdefault(case_id, set()).add(slice_index)
    protected_backgrounds = [
        record
        for record in backgrounds
        if any(
            abs(record[3] - positive_index) <= neighbor_radius
            for positive_index in positive_indices.get(record[2], set())
        )
    ]
    protected_labels = {record[1] for record in protected_backgrounds}
    optional_backgrounds = [
        record for record in backgrounds if record[1] not in protected_labels
    ]
    target_backgrounds = max(
        len(protected_backgrounds),
        round(len(selected_positives) * background_to_positive_ratio),
    )
    optional_count = min(
        max(0, target_backgrounds - len(protected_backgrounds)),
        len(optional_backgrounds),
    )
    selected_backgrounds = protected_backgrounds + rng.sample(
        optional_backgrounds, optional_count
    )

    selected_labels = {
        record[1] for record in selected_positives + selected_backgrounds
    }
    archive_image_dir.mkdir(parents=True, exist_ok=True)
    archive_label_dir.mkdir(parents=True, exist_ok=True)
    excluded = 0
    for image_path, label_path, _, _, _ in records:
        if label_path in selected_labels:
            continue
        shutil.move(str(image_path), str(archive_image_dir / image_path.name))
        shutil.move(str(label_path), str(archive_label_dir / label_path.name))
        excluded += 1

    # One repeat is a moderate correction for the rarest class I (nc=5: I=1).
    oversample_rare_classes(base_dir, {1: 1})

    for cache_path in (base_dir / "labels").glob("*.cache"):
        cache_path.unlink()

    final_labels = list(label_dir.glob("*.txt"))
    summary = {
        "original_positive_slices": len(selected_positives),
        "background_slices": len(selected_backgrounds),
        "class_I_1_slices_before_repeat": presence[1],
        "class_P_2_slices": presence[2],
        "class_S_3_slices": presence[3],
        "class_U_4_slices": presence[4],
        "excluded_slices": excluded,
        "final_training_slices": len(final_labels),
    }
    logger_processing_manager.info(f"Class balancing completed: {summary}")
    return summary


def create_train_set(reset: bool = True) -> None:
    """
    Prepares the training dataset by creating necessary directories,
    copying and renaming files, and splitting them into finetune, validation, and test sets.
    """

    logger_processing_manager.info(
        f"Creating finetune set in {settings.train_set_path} from {settings.processed_data_path}"
    )

    try:
        if reset and settings.train_set_path.exists():
            shutil.rmtree(settings.train_set_path)
        create_dirs(settings.train_set_path, ['images', 'labels'])
        move_and_rename_files(settings.processed_data_path, settings.train_set_path)
        split_dataset(settings.train_set_path, train_ratio=0.8, val_ratio=0.2)
        # Уровень баланса частых классов (P/S) настраивается через env для экспериментов.
        # Меньше => жёстче баланс (сильнее режем P/S к уровню редких I/U).
        target_common = int(os.getenv("FINETUNE_BALANCE_COMMON", "800"))
        balance_training_classes(settings.train_set_path, target_common_slices=target_common)
        logger_processing_manager.info(
            f"Train set creation completed (target_common_slices={target_common})."
        )
    except Exception as e:
        logger_processing_manager.error(f"Failed to create finetune set: {str(e)}")
        raise


if __name__ == '__main__':
    create_train_set()
