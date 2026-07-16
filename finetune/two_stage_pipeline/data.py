from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


SOURCE_CLASS_IDS = {1, 2, 3, 4}
MEDICAL_CLASSES = {1: "I", 2: "P", 3: "S", 4: "U"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Annotation:
    class_id: int
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CropRecord:
    split: str
    crop_path: str
    source_image: str
    source_label: str
    source_class_id: int | None
    class_name: str
    object_index: int | None
    x1: int
    y1: int
    x2: int
    y2: int
    generated_by: str


def parse_yolo_segmentation(path: Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(
                f"{path}:{line_number}: expected class ID and at least "
                "three x/y polygon points"
            )
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: non-numeric YOLO annotation"
            ) from exc
        if class_id not in SOURCE_CLASS_IDS:
            raise ValueError(
                f"{path}:{line_number}: expected source class 1..4, "
                f"got {class_id}"
            )
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in coordinates):
            raise ValueError(
                f"{path}:{line_number}: polygon coordinates must be finite "
                "and normalized to [0, 1]"
            )
        points = tuple(zip(coordinates[0::2], coordinates[1::2]))
        annotations.append(Annotation(class_id=class_id, points=points))
    return annotations


def annotation_box(
    annotation: Annotation,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    xs = [point[0] * width for point in annotation.points]
    ys = [point[1] * height for point in annotation.points]
    return min(xs), min(ys), max(xs), max(ys)


def contextual_square_box(
    box: Sequence[float],
    width: int,
    height: int,
    context: float = 2.0,
    minimum_side: int = 48,
) -> tuple[int, int, int, int]:
    if context < 1:
        raise ValueError("context must be at least 1.0")
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 1.0) * context
    side = min(max(side, minimum_side), width, height)

    left = min(max(center_x - side / 2, 0), width - side)
    top = min(max(center_y - side / 2, 0), height - side)
    right = left + side
    bottom = top + side
    return (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(right)),
        int(math.ceil(bottom)),
    )


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def find_image(image_dir: Path, stem: str) -> Path:
    matches = [
        image_dir / f"{stem}{extension}"
        for extension in IMAGE_EXTENSIONS
        if (image_dir / f"{stem}{extension}").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one image for {stem} in {image_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            # Ultralytics may initialize its settings directory before the
            # generated dataset exists. That service-only directory must not
            # make the first data build look like an existing dataset.
            unexpected_entries = [
                path for path in output_root.iterdir() if path.name != "ultralytics"
            ]
            if unexpected_entries:
                raise FileExistsError(
                    f"{output_root} already contains generated data; "
                    "pass --overwrite to rebuild it"
                )
            return
        resolved = output_root.resolve()
        workspace = Path(__file__).resolve().parents[2]
        if workspace not in resolved.parents:
            raise ValueError(f"Refusing to remove path outside {workspace}: {resolved}")
        shutil.rmtree(resolved)
    output_root.mkdir(parents=True)


def _write_detector_yaml(detector_root: Path) -> Path:
    yaml_path = detector_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {detector_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: lesion\n",
        encoding="utf-8",
    )
    return yaml_path


def _binary_label_text(label_path: Path) -> str:
    output: list[str] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = raw_line.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"{label_path}:{line_number}: invalid class ID"
            ) from exc
        if class_id not in SOURCE_CLASS_IDS:
            raise ValueError(
                f"{label_path}:{line_number}: expected source class 1..4, "
                f"got {class_id}"
            )
        output.append(" ".join(["0", *fields[1:]]))
    return "\n".join(output)


def _save_crop(
    image: Image.Image,
    destination: Path,
    box: tuple[int, int, int, int],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).convert("RGB").save(destination, format="PNG")


def _random_background_records(
    *,
    split: str,
    pairs: list[tuple[Path, Path]],
    target_count: int,
    output_root: Path,
    candidate_sides: list[int],
    minimum_side: int,
    seed: int,
) -> list[CropRecord]:
    if target_count <= 0:
        return []
    rng = random.Random(seed)
    prepared: list[
        tuple[
            Path,
            Path,
            int,
            int,
            list[tuple[float, float, float, float]],
        ]
    ] = []
    for image_path, label_path in pairs:
        with Image.open(image_path) as image:
            boxes = [
                annotation_box(annotation, image.width, image.height)
                for annotation in parse_yolo_segmentation(label_path)
            ]
            prepared.append(
                (image_path, label_path, image.width, image.height, boxes)
            )

    records: list[CropRecord] = []
    attempts = 0
    maximum_attempts = max(100, target_count * 100)
    while len(records) < target_count and attempts < maximum_attempts:
        attempts += 1
        image_path, label_path, width, height, lesion_boxes = rng.choice(prepared)
        max_side = min(width, height)
        raw_side = rng.choice(candidate_sides) if candidate_sides else minimum_side
        side = min(max(raw_side, minimum_side), max_side)
        left = rng.randint(0, max(0, width - side))
        top = rng.randint(0, max(0, height - side))
        crop_box = (left, top, left + side, top + side)
        if any(box_iou(crop_box, lesion_box) > 0.01 for lesion_box in lesion_boxes):
            continue

        index = len(records)
        destination = (
            output_root
            / "classifier"
            / "images"
            / split
            / "background"
            / f"{image_path.stem}__background_{index:05d}.png"
        )
        with Image.open(image_path) as image:
            _save_crop(image, destination, crop_box)
        records.append(
            CropRecord(
                split=split,
                crop_path=str(destination),
                source_image=str(image_path),
                source_label=str(label_path),
                source_class_id=None,
                class_name="background",
                object_index=None,
                x1=crop_box[0],
                y1=crop_box[1],
                x2=crop_box[2],
                y2=crop_box[3],
                generated_by="random_background",
            )
        )
    if len(records) != target_count:
        raise RuntimeError(
            f"Generated only {len(records)} of {target_count} background crops "
            f"for {split}; source images may be too densely annotated"
        )
    return records


def _resolve_split_sources(
    source_root: Path,
    *,
    full: bool,
) -> dict[str, list[tuple[Path, Path]]]:
    """
    Map each target split to its (images_dir, labels_dir) source directories.

    Legacy mode (``full=False``) reproduces the original single-source build:
    train from ``images/labels/finetune`` and validation from ``.../val``.

    Full mode (``full=True``) additionally merges the ``excluded_balance``
    directory into the train split. That directory holds the 7 480 slices an
    earlier multiclass balancing step removed; they are legitimate lesion
    examples for the binary task. Validation is never extended.
    """
    train_sources = [
        (source_root / "images" / "finetune", source_root / "labels" / "finetune"),
    ]
    if full:
        excluded = source_root / "excluded_balance"
        train_sources.append((excluded / "images", excluded / "labels"))
    return {
        "train": train_sources,
        "val": [(source_root / "images" / "val", source_root / "labels" / "val")],
    }


def _collect_split_pairs(
    sources: list[tuple[Path, Path]],
    *,
    split: str,
    exclude_repeats: bool,
) -> list[tuple[Path, Path]]:
    """
    Gather (image, label) pairs for one split across its source directories.

    ``__repeat_`` copies are dropped when ``exclude_repeats`` is set, and a
    basename appearing in more than one source is a fatal ambiguity.
    """
    pairs: list[tuple[Path, Path]] = []
    seen: dict[str, Path] = {}
    for images_dir, labels_dir in sources:
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Missing source split directories: {images_dir}, {labels_dir}"
            )
        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = label_path.stem
            if exclude_repeats and "__repeat_" in stem:
                continue
            if stem in seen:
                raise ValueError(
                    f"Duplicate basename {stem!r} in split {split!r}: "
                    f"{seen[stem]} and {label_path}"
                )
            seen[stem] = label_path
            pairs.append((find_image(images_dir, stem), label_path))
    return pairs


def prepare_two_stage_data(
    source_root: Path,
    output_root: Path,
    *,
    context: float = 2.0,
    minimum_side: int = 48,
    train_background_ratio: float = 1.0,
    val_background_ratio: float = 1.0,
    seed: int = 42,
    overwrite: bool = False,
    full: bool = False,
    exclude_repeats: bool | None = None,
) -> dict[str, object]:
    """
    Build isolated binary-detector and crop-classifier datasets.

    The source study-level train/validation split is preserved exactly.
    Source images and labels are never modified.

    ``full`` merges ``excluded_balance`` into the train split; when it is left
    to the default, ``exclude_repeats`` follows ``full`` (full builds drop the
    ``__repeat_`` copies, legacy builds keep them for byte-for-byte parity).
    """
    if train_background_ratio < 0 or val_background_ratio < 0:
        raise ValueError("background ratios cannot be negative")
    if exclude_repeats is None:
        exclude_repeats = full
    _prepare_output(output_root, overwrite)
    detector_root = output_root / "detector"
    classifier_root = output_root / "classifier" / "images"
    records: list[CropRecord] = []

    split_sources = _resolve_split_sources(source_root, full=full)
    collected: dict[str, list[tuple[Path, Path]]] = {
        target_split: _collect_split_pairs(
            sources, split=target_split, exclude_repeats=exclude_repeats
        )
        for target_split, sources in split_sources.items()
    }
    train_stems = {label.stem for _, label in collected.get("train", [])}
    val_stems = {label.stem for _, label in collected.get("val", [])}
    leaked = train_stems & val_stems
    if leaked:
        example = ", ".join(sorted(leaked)[:5])
        raise ValueError(
            f"train/val leakage: {len(leaked)} shared basenames (e.g. {example})"
        )

    summary: dict[str, object] = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "context": context,
        "minimum_side": minimum_side,
        "full": full,
        "exclude_repeats": exclude_repeats,
        "sources": {
            target_split: [str(labels.resolve()) for _, labels in sources]
            for target_split, sources in split_sources.items()
        },
        "splits": {},
    }

    for target_split in ("train", "val"):
        pairs = collected[target_split]
        positive_count = 0
        candidate_sides: list[int] = []
        class_counts = {class_name: 0 for class_name in MEDICAL_CLASSES.values()}

        for image_path, label_path in pairs:
            detector_image = detector_root / "images" / target_split / image_path.name
            detector_label = detector_root / "labels" / target_split / label_path.name
            _link_or_copy(image_path, detector_image)
            detector_label.parent.mkdir(parents=True, exist_ok=True)
            detector_label.write_text(
                _binary_label_text(label_path),
                encoding="utf-8",
            )

            annotations = parse_yolo_segmentation(label_path)
            if not annotations:
                continue
            with Image.open(image_path) as image:
                for object_index, annotation in enumerate(annotations):
                    raw_box = annotation_box(annotation, image.width, image.height)
                    crop_box = contextual_square_box(
                        raw_box,
                        image.width,
                        image.height,
                        context=context,
                        minimum_side=minimum_side,
                    )
                    class_name = MEDICAL_CLASSES[annotation.class_id]
                    destination = (
                        classifier_root
                        / target_split
                        / class_name
                        / f"{image_path.stem}__object_{object_index:04d}.png"
                    )
                    _save_crop(image, destination, crop_box)
                    candidate_sides.append(crop_box[2] - crop_box[0])
                    positive_count += 1
                    class_counts[class_name] += 1
                    records.append(
                        CropRecord(
                            split=target_split,
                            crop_path=str(destination),
                            source_image=str(image_path),
                            source_label=str(label_path),
                            source_class_id=annotation.class_id,
                            class_name=class_name,
                            object_index=object_index,
                            x1=crop_box[0],
                            y1=crop_box[1],
                            x2=crop_box[2],
                            y2=crop_box[3],
                            generated_by="ground_truth",
                        )
                    )

        background_ratio = (
            train_background_ratio if target_split == "train" else val_background_ratio
        )
        background_count = round(positive_count * background_ratio)
        background_records = _random_background_records(
            split=target_split,
            pairs=pairs,
            target_count=background_count,
            output_root=output_root,
            candidate_sides=candidate_sides,
            minimum_side=minimum_side,
            seed=seed + (0 if target_split == "train" else 1),
        )
        records.extend(background_records)
        summary["splits"][target_split] = {
            "source_images": len(pairs),
            "positive_crops": positive_count,
            "background_crops": len(background_records),
            "class_counts": class_counts,
        }

    detector_yaml = _write_detector_yaml(detector_root)
    manifest_path = output_root / "classifier" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [field.name for field in fields(CropRecord)]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    summary["detector_yaml"] = str(detector_yaml)
    summary["classifier_root"] = str(classifier_root)
    summary_path = output_root / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Prepare isolated datasets for the two-stage lesion pipeline."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "finetune" / "fileStorage" / "train_set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to two_stage_full when --full is "
        "set, otherwise two_stage.",
    )
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--minimum-side", type=int, default=48)
    parser.add_argument("--train-background-ratio", type=float, default=1.0)
    parser.add_argument("--val-background-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Merge excluded_balance into train and drop __repeat_ copies.",
    )
    parser.add_argument(
        "--keep-repeats",
        dest="exclude_repeats",
        action="store_false",
        default=None,
        help="Keep __repeat_ copies even in a --full build.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output = args.output
    if output is None:
        dataset_name = "two_stage_full" if args.full else "two_stage"
        output = project_root / "finetune" / "fileStorage" / dataset_name
    summary = prepare_two_stage_data(
        args.source,
        output,
        context=args.context,
        minimum_side=args.minimum_side,
        train_background_ratio=args.train_background_ratio,
        val_background_ratio=args.val_background_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
        full=args.full,
        exclude_repeats=args.exclude_repeats,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
