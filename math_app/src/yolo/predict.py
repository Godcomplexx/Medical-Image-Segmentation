import os
import cv2
import numpy as np
from pathlib import Path

from cv2 import Mat
from numpy import ndarray, dtype, unsignedinteger
from numpy._typing import _8Bit
from ultralytics import YOLO
from typing import Optional, Tuple, Any
from config.logs_config import logger_main
from config.settings import settings
import nibabel as nib
from config.labeling_config import labeling


transpose_dict = {
    1 : 'I',
    2 : 'P',
    3 : 'S',
    4 : 'U',
}
transpose_dict_ = {
    'I' : 0,
    'P' : 1,
    'S' : 2,
    'U' : 3,
}
def example_case(
    path_to_image: Path,
    path_to_model: Path,
    save_image_dir: Optional[Path] = None,
    to_mask: Optional[Path] = None
) -> tuple[None, None, None] | tuple[ndarray, ndarray, ndarray]:
    """
    Perform object detection using YOLO, save images with bounding boxes and optional masks.

    Args:
        path_to_image (Path): Path to the image file.
        path_to_model (Path): Path to the YOLO model.
        save_image_dir (Optional[Path]): Directory to save images with bounding boxes.
        to_mask (Optional[Path]): Directory to save the mask images.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: The original image and the overlayed mask, if detection is successful.
    """
    try:
        image_path = path_to_image.as_posix()
        model_path = path_to_model.as_posix()

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        h, w, _ = img.shape

        model = YOLO(model_path)
        results = model.predict(img, imgsz=640, box=True, show_labels=True, line_width=2)

        detected_classes = []
        overlay = np.zeros_like(img, dtype=np.uint8)
        label_map = np.zeros((h, w), dtype=np.uint8)
        base_filename = path_to_image.stem
        mask_dict = {}
        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue

            boxes = result.boxes
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None
            class_names = result.names

            for j, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
                class_index = int(box.cls.item())
                class_name = class_names[class_index]
                detected_classes.append((class_name, (x_min, y_min, x_max, y_max)))

                mask = masks[j] * 255 if masks is not None else None
                if mask is not None:
                    mask = cv2.resize(mask.astype(np.uint8), (w, h))

                    # The model uses zero-based IDs, while zero in the
                    # semantic volume is reserved for background.
                    label_map[mask > 127] = class_index + 1

                    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

                    for class_id, color in transpose_dict.items():
                        for i in range(3):  # по каналам B, G, R
                            colored_mask[:, :, i][label_map == class_id] = labeling.color_dict[color][i]


                    overlay = cv2.add(overlay, colored_mask.astype(np.uint8))

                    if to_mask:

                        key =f"{base_filename}_{class_name}"
                        if key in  mask_dict:
                            mask_dict[key] = cv2.bitwise_or(mask_dict[key], mask)
                        else:
                            mask_dict[key] = mask
                        # cv2.imwrite(str(to_mask / f"{base_filename}_{class_name}.png"), mask)

            to_mask.mkdir(parents=True, exist_ok=True)
            for name, mask in mask_dict.items():
                out_path = to_mask / f"{name}.png"
                cv2.imwrite(str(out_path), mask)



        # result_image = cv2.addWeighted(img, 0.8, overlay, 0.5, 0)

        # # Отрисовка прямоугольников и меток
        # for class_name, (x_min, y_min, x_max, y_max) in detected_classes:
        #     cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #     cv2.putText(result_image, transpose_dict[int(class_name)], (x_min, y_min - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # if save_image_dir:
        #     save_image_dir.mkdir(parents=True, exist_ok=True)
        #     output_image_path = save_image_dir / f"{base_filename}.png"
        #     cv2.imwrite(str(output_image_path), result_image)

        return img, overlay, label_map

    except FileNotFoundError as e:
        logger_main.error(f"File not found: {e}")
    except cv2.error as e:
        logger_main.error(f"OpenCV error: {e}")
    except Exception as e:
        logger_main.error(f"An unexpected error occurred: {e}")

def predict_on_folder(path2t2: Path):

    final_path: Path = path2t2 / 'Yolo' / 'images'
    path2mask: Path = path2t2 / 'Mask'
    path2predict: Path = path2t2 / 'Predict'
    semantic_mask = []

    nii = nib.load(path2t2 / 'T2.nii')
    affine = nii.affine
    header = nii.header
    shape = nii.shape
    iterable = sorted(final_path.glob("*"),key= lambda a: int(a.stem.split('_')[-1]))
    for image_path in iterable:
        _,_,label_mp = example_case(image_path, settings.path_to_model, save_image_dir=path2predict, to_mask=path2mask)
        semantic_mask.append(label_mp)
    target_shape = (shape[0], shape[1])
    semantic_mask = np.array(semantic_mask)
    resized_slices = []
    for i in range(semantic_mask.shape[0]):
        resized = cv2.resize(semantic_mask[i,:,:], target_shape[::-1], interpolation=cv2.INTER_NEAREST)
        resized_slices.append(resized)

    resized_volume = np.stack(resized_slices, axis=0)
    resized_volume = np.transpose(resized_volume, (1, 2, 0))
    seg_nii = nib.Nifti1Image(resized_volume.astype(np.uint8), affine)
    path2predict.mkdir(exist_ok=True)
    nib.save(seg_nii, path2predict / "segmentation.nii")
    import shutil
    shutil.rmtree(final_path.parent)



