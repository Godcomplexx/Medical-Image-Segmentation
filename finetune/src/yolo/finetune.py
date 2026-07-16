import csv
import os
from pathlib import Path
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parents[2] / "fileStorage" / "ultralytics"),
)

import torch
from ultralytics import YOLO
from finetune.config.logs_config import *


# nc=5 схема: реальные классы 1..4 (класс 0 не используется).
MEDICAL_CLASS_NAMES = {1: "I", 2: "P", 3: "S", 4: "U"}


def _box_checkpoint_callback(target_recall: float):
    """Save the checkpoint with the strongest balanced box performance."""
    state = {"best_score": float("-inf")}

    def on_model_save(trainer) -> None:
        box = trainer.validator.metrics.box
        class_rows = []
        recalls = {class_id: 0.0 for class_id in MEDICAL_CLASS_NAMES}
        for metric_index, class_id in enumerate(box.ap_class_index):
            precision, recall, map50, map50_95 = box.class_result(metric_index)
            class_id = int(class_id)
            recalls[class_id] = float(recall)
            class_rows.append(
                {
                    "epoch": trainer.epoch + 1,
                    "class_id": class_id,
                    "class_name": MEDICAL_CLASS_NAMES[class_id],
                    "precision": float(precision),
                    "recall": float(recall),
                    "map50": float(map50),
                    "map50_95": float(map50_95),
                    "target_recall_met": float(recall) >= target_recall,
                }
            )

        metrics_path = Path(trainer.save_dir) / "box_metrics.csv"
        write_header = not metrics_path.exists()
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            fieldnames = [
                "epoch",
                "class_id",
                "class_name",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "target_recall_met",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(class_rows)

        minimum_recall = min(recalls.values())
        mean_recall = sum(recalls.values()) / len(recalls)
        score = (
            0.6 * minimum_recall
            + 0.2 * mean_recall
            + 0.1 * float(box.map50)
            + 0.1 * float(box.map)
        )
        if score > state["best_score"]:
            state["best_score"] = score
            best_box = Path(trainer.save_dir) / "weights" / "best_box.pt"
            shutil.copy2(trainer.last, best_box)
            logger_main.info(
                "Saved best_box.pt at epoch %d: score=%.5f, recalls=%s",
                trainer.epoch + 1,
                score,
                {
                    MEDICAL_CLASS_NAMES[class_id]: round(recall, 4)
                    for class_id, recall in recalls.items()
                },
            )

    return on_model_save


def finetune():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training was not started to avoid an "
            "unexpectedly slow CPU run."
        )

    epochs = int(os.getenv("FINETUNE_EPOCHS", "100"))
    batch = int(os.getenv("FINETUNE_BATCH", "12"))
    image_size = int(os.getenv("FINETUNE_IMGSZ", "640"))
    mask_ratio = int(os.getenv("FINETUNE_MASK_RATIO", "2"))
    workers = int(os.getenv("FINETUNE_TRAIN_WORKERS", "2"))
    learning_rate = float(os.getenv("FINETUNE_LR0", "0.001"))
    patience = int(os.getenv("FINETUNE_PATIENCE", "25"))
    target_box_recall = float(os.getenv("FINETUNE_TARGET_BOX_RECALL", "0.8"))
    device = os.getenv("FINETUNE_DEVICE", "0")
    run_name = os.getenv("FINETUNE_RUN_NAME", "flair_v8_box_balanced")
    # Рецептные рычаги, вынесенные в env для A/B без правки кода.
    optimizer = os.getenv("FINETUNE_OPTIMIZER", "AdamW")  # AdamW | SGD | auto
    mosaic = float(os.getenv("FINETUNE_MOSAIC", "1.0"))
    close_mosaic = int(os.getenv("FINETUNE_CLOSE_MOSAIC", "10"))
    warmup_epochs = float(os.getenv("FINETUNE_WARMUP", "3"))
    lrf = float(os.getenv("FINETUNE_LRF", "0.1"))
    project_dir = settings.database_path / "training_runs"
    # Может быть путём к локальному .pt ЛИБО именем модели для авто-загрузки
    # (например, yolo11m-seg.pt / yolov8m-seg.pt — чистый старт от COCO).
    model_arg = os.getenv("FINETUNE_MODEL", str(settings.path_to_model))

    logger_main.info(
        "Starting YOLO training: "
        f"model={model_arg}, GPU={torch.cuda.get_device_name(0)}, epochs={epochs}, "
        f"batch={batch}, imgsz={image_size}, mask_ratio={mask_ratio}, "
        f"workers={workers}, optimizer={optimizer}, lr0={learning_rate}, "
        f"mosaic={mosaic}, close_mosaic={close_mosaic}, warmup={warmup_epochs}"
    )

    model = YOLO(model_arg)
    model.add_callback(
        "on_model_save",
        _box_checkpoint_callback(target_box_recall),
    )

    results = model.train(
        data=settings.path_to_yolo_config,
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        workers=workers,
        device=device,
        project=project_dir,
        name=run_name,
        patience=patience,
        seed=42,
        deterministic=True,
        cache=False,
        amp=True,
        cos_lr=True,
        mask_ratio=mask_ratio,
        overlap_mask=True,
        save_period=5,
        plots=True,
        optimizer=optimizer,
        lr0=learning_rate,
        lrf=lrf,
        warmup_epochs=warmup_epochs,
        mosaic=mosaic,
        close_mosaic=close_mosaic,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.05,
        degrees=0.0,
        translate=0.05,
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
    )
    save_dir = getattr(model.trainer, "save_dir", project_dir / run_name)
    logger_main.info(f"Training artifacts saved to {save_dir}")
    return results


if __name__ == '__main__':
    finetune()
