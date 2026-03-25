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

        create_train_set()

        write_config_yaml()

        write_hyp_yaml()

        finetune()

        return True
    except Exception as e:
        logger_processing_manager.error(f"Error: {str(e)}")
        return False


if __name__ == '__main__':
    run_pipeline()
