from pathlib import Path

from finetune.src.pipelines.yolo_dataset import (
    PIPELINE_VERSION,
    is_current_completion_marker,
)


def test_legacy_empty_completion_marker_is_stale(tmp_path: Path) -> None:
    marker = tmp_path / ".complete"
    marker.touch()

    assert not is_current_completion_marker(marker)


def test_versioned_completion_marker_is_current(tmp_path: Path) -> None:
    marker = tmp_path / ".complete"
    marker.write_text(PIPELINE_VERSION, encoding="utf-8")

    assert is_current_completion_marker(marker)
