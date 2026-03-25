from pathlib import Path, PurePath
from typing import Union
from pydantic_settings import BaseSettings


class Base(BaseSettings):

    # General
    project_dir: Path = Path(__file__).resolve().parent.parent
    database_path: Path = project_dir / "fileStorage"

    # Input/Output
    input_data_path: Path = database_path / "input_data"
    processed_data_path: Path = database_path / "processed_data"

    # FastSurfer related paths
    morphometry_path: Path = database_path / "for_volume_calc"
    lic_path: Path = project_dir / "config" / "fast_surfer_lic" / "license.txt"
    color_settings: Path = database_path / "color_set" / "FreeSurferColorLUT.txt"

    # Logging
    log_level: str = "DEBUG"
    logs_path: Path = project_dir / "logs"

    # Yolo related paths
    train_set_path: Path = database_path / "train_set"
    # train_img_path: Path = train_set_path / 'train'
    # val_img_path: Path = train_set_path / 'val'
    path_to_model: Path = project_dir / "best.pt"
    path_to_yolo_config: Path = project_dir / "yolo_config" / "config.yaml"
    path_to_yolo_hyper_params: Path = project_dir / "yolo_config" / "hyp.yaml"
    # prediction_path: Path = project_dir / "fileStorage" / "predict"



    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_dirs()

    def _create_dirs(self):
        for path in [
            self.processed_data_path,
            self.morphometry_path,
            self.logs_path,
            self.train_set_path,
            self.input_data_path

        ]:
            path.mkdir(parents=True, exist_ok=True)

    # Yolo hyperparameters
    yolo_hyp: dict[str, float] = {
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    }
    yolo_config: dict[str, Union[str, list[str]]] = {
        "path": str(train_set_path),
        "train": str(train_set_path / "images" / "train"),
        "val": str(train_set_path / "images" / "val"),
        "names": ['0', '1', '2', '3', '4'],
    }

    class Config:
        env_file: str = f"{Path(__file__).resolve().parent.parent}/.env"
        env_file_encoding: str = 'utf-8'
        extra: str = "allow"


settings = Base()
