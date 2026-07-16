from __future__ import annotations

from pathlib import Path

from finetune.config.logs_config import logger_main
from finetune.config.settings import settings


SOURCE_IDS = {1, 2, 3, 4}
TARGET_IDS = {0, 1, 2, 3}


def remap_label_tree(label_root: Path) -> tuple[int, int]:
    """Atomically convert YOLO class IDs 1..4 to the required 0..3."""
    label_paths = sorted(label_root.rglob("*.txt"))
    parsed: dict[Path, list[list[str]]] = {}
    observed_ids: set[int] = set()

    for label_path in label_paths:
        rows: list[list[str]] = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            fields = line.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError as exc:
                raise ValueError(
                    f"{label_path}:{line_number}: invalid class ID"
                ) from exc
            observed_ids.add(class_id)
            rows.append(fields)
        parsed[label_path] = rows

    if not observed_ids:
        raise ValueError(f"No annotations found under {label_root}")
    if 0 in observed_ids and 4 in observed_ids:
        raise ValueError("Mixed zero-based and one-based labels detected")
    if 0 in observed_ids:
        if not observed_ids <= TARGET_IDS:
            raise ValueError(f"Unexpected class IDs: {sorted(observed_ids)}")
        logger_main.info("YOLO labels are already zero-based")
        return 0, sum(len(rows) for rows in parsed.values())
    if 4 not in observed_ids:
        raise ValueError(
            "Class IDs are ambiguous: expected either class 0 or class 4"
        )
    if not observed_ids <= SOURCE_IDS:
        raise ValueError(f"Unexpected class IDs: {sorted(observed_ids)}")

    annotation_count = 0
    changed_files = 0
    for label_path, rows in parsed.items():
        if not rows:
            continue
        remapped = [" ".join([str(int(fields[0]) - 1), *fields[1:]]) for fields in rows]
        temporary_path = label_path.with_suffix(".txt.tmp")
        temporary_path.write_text("\n".join(remapped), encoding="utf-8")
        temporary_path.replace(label_path)
        changed_files += 1
        annotation_count += len(rows)

    for cache_path in label_root.parent.rglob("*.cache"):
        cache_path.unlink()

    logger_main.info(
        "Remapped %d annotations in %d label files from 1..4 to 0..3",
        annotation_count,
        changed_files,
    )
    return changed_files, annotation_count


if __name__ == "__main__":
    remap_label_tree(settings.train_set_path / "labels")
