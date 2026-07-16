"""Turn U-Net masks into lesion candidate boxes for the anatomy classifier."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import imread_rgb
from .unet import UNet


class LesionSegmenter:
    def __init__(
        self,
        weights: Path,
        *,
        imgsz: int = 640,
        threshold: float = 0.5,
        min_area: int = 4,
        device: str = "cuda:0",
        encoder: str | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.threshold = threshold
        self.min_area = min_area
        if encoder:
            # Must match the architecture the weights were trained with
            # (train_seg builds smp.Unet when --encoder is given).
            import segmentation_models_pytorch as smp

            self.model = smp.Unet(
                encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1
            ).to(self.device)
        else:
            self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        state = torch.load(str(weights), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def probability_map(self, image_rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
        height0, width0 = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            logits = self.model(tensor)
        prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        return prob, width0, height0

    def boxes(
        self,
        image_path: Path,
        fp_filter=None,
        fp_threshold: float = 0.5,
    ) -> list[dict[str, object]]:
        """
        Candidate boxes in original-slice pixel coordinates, with a mask score.

        When ``fp_filter`` is given, each blob is scored by the appearance-based
        lesion/not-lesion model and dropped below ``fp_threshold`` — the precision
        gate the anatomy classifier cannot provide.
        """
        image = imread_rgb(image_path)
        prob, width0, height0 = self.probability_map(image)
        mask = (prob > self.threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rgb_resized = (
            cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if fp_filter is not None
            else None
        )
        scale_x = width0 / self.imgsz
        scale_y = height0 / self.imgsz
        candidates: list[dict[str, object]] = []
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < self.min_area:
                continue
            comp = labels == component
            if fp_filter is not None:
                if fp_filter.is_lesion_prob(prob, rgb_resized, comp) < fp_threshold:
                    continue
            x = stats[component, cv2.CC_STAT_LEFT]
            y = stats[component, cv2.CC_STAT_TOP]
            w = stats[component, cv2.CC_STAT_WIDTH]
            h = stats[component, cv2.CC_STAT_HEIGHT]
            score = float(prob[comp].mean())
            candidates.append(
                {
                    "box_xyxy": [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y],
                    "score": score,
                }
            )
        return candidates
