from finetune.src.yolo.train_test_split import create_train_set
from finetune.src.yolo.config_writer import write_config_yaml, write_hyp_yaml
from finetune.src.yolo.finetune import finetune
from finetune.config.logs_config import *


def run_pipeline() -> bool:
    """
    Creates yolo dataset (images and labels) from DICOM and Markups
    :return:
    """
    try:
        logger_main.info(f"Processing")

        train_images = settings.train_set_path / "images" / "finetune"
        val_images = settings.train_set_path / "images" / "val"
        dataset_ready = (
            train_images.exists()
            and val_images.exists()
            and any(train_images.glob("*.png"))
            and any(val_images.glob("*.png"))
        )
        if dataset_ready:
            logger_main.info("Using the existing prepared train/val dataset.")
        else:
            create_train_set(reset=True)

        write_config_yaml()

        write_hyp_yaml()

        finetune()

        return True
    except Exception as e:
        logger_processing_manager.error(f"Error: {str(e)}")
        return False


if __name__ == '__main__':
    run_pipeline()
