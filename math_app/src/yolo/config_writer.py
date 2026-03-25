from pathlib import Path
import yaml
from config.logs_config import *


def write_config_yaml(
    output_path: Path = settings.path_to_yolo_config
) -> None:
    config = settings.yolo_config
    with open(output_path, "w") as f:
        yaml.dump(config, f)
    logger_main.info(f"YOLO dataset config written to {output_path}")


def write_hyp_yaml(
    output_path: Path = settings.path_to_yolo_hyper_params
) -> None:
    hyp = settings.yolo_hyp
    with open(output_path, "w") as f:
        yaml.dump(hyp, f)
    logger_main.info(f"YOLO hyp config written to {output_path}")


if __name__ == '__main__':
    write_config_yaml()
    write_hyp_yaml()
