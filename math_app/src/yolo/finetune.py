import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from ultralytics import YOLO
from config.logs_config import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

logger_main.info(torch.cuda.device_count())
logger_main.info(torch.cuda.get_device_name(0))


def finetune():
    model = YOLO(settings.path_to_model)

    model.train(
        data=settings.path_to_yolo_config,
        epochs=100,
        imgsz=640
    )
    model.save("trained_yolo.pt")


if __name__ == '__main__':
    finetune()
