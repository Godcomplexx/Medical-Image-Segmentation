from pathlib import Path
from types import SimpleNamespace

from finetune.src.yolo.finetune import _box_checkpoint_callback


class _BoxMetrics:
    ap_class_index = [0, 1, 2, 3]
    map50 = 0.5
    map = 0.25

    @staticmethod
    def class_result(index: int) -> tuple[float, float, float, float]:
        recall = [0.81, 0.82, 0.83, 0.84][index]
        return 0.75, recall, 0.5, 0.25


def test_box_callback_saves_balanced_checkpoint_and_class_report(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    last = weights / "last.pt"
    last.write_bytes(b"checkpoint")
    trainer = SimpleNamespace(
        epoch=4,
        save_dir=tmp_path,
        last=last,
        validator=SimpleNamespace(
            metrics=SimpleNamespace(box=_BoxMetrics())
        ),
    )

    callback = _box_checkpoint_callback(target_recall=0.8)
    callback(trainer)

    assert (weights / "best_box.pt").read_bytes() == b"checkpoint"
    report = (tmp_path / "box_metrics.csv").read_text(encoding="utf-8")
    assert report.count("\n") == 5
    assert "5,0,I,0.75,0.81,0.5,0.25,True" in report
    assert "5,3,U,0.75,0.84,0.5,0.25,True" in report
