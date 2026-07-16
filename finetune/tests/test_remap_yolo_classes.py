from pathlib import Path

from finetune.src.tools.remap_yolo_classes import remap_label_tree


def test_remap_label_tree_converts_one_based_ids_and_is_idempotent(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    label = labels / "slice_1_case.txt"
    label.write_text(
        "1 0.1 0.1 0.2 0.1 0.2 0.2\n"
        "4 0.3 0.3 0.4 0.3 0.4 0.4\n",
        encoding="utf-8",
    )
    (labels / "empty.txt").write_text("", encoding="utf-8")

    assert remap_label_tree(labels) == (1, 2)
    assert label.read_text(encoding="utf-8").splitlines() == [
        "0 0.1 0.1 0.2 0.1 0.2 0.2",
        "3 0.3 0.3 0.4 0.3 0.4 0.4",
    ]
    assert remap_label_tree(labels) == (0, 2)
