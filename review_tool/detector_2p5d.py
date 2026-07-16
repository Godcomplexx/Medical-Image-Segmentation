# -*- coding: utf-8 -*-
"""
Инференс нашего 2.5D-детектора очагов (smp U-Net, ResNet34) на объёме FLAIR.

Детектор обучался на 2.5D-входе (соседние срезы z-1,z,z+1 как RGB-каналы) поверх
нормализации normalize_mri_volume. Здесь тот же конвейер применяется к произвольному
объёму FLAIR и возвращается бинарная маска очагов (та же сетка вокселей, что и вход).
Нужен, чтобы дополнить находки nnU-Net (union) и сократить пропуски — особенно
мелкие очаги, которые объёмная модель теряет.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import nibabel as nib

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from finetune.src.yolo.image_generator import normalize_mri_volume
from finetune.two_stage_pipeline.segmentation.predict_seg import LesionSegmenter

DEFAULT_WEIGHTS = (REPO / "finetune" / "fileStorage" / "two_stage_full" /
                   "seg_runs" / "unet_2p5d_resnet34_completed2" / "weights" / "best.pt")


class Detector2p5d:
    def __init__(self, weights=DEFAULT_WEIGHTS, encoder="resnet34",
                 imgsz=640, threshold=0.5, device="cuda:0"):
        self.seg = LesionSegmenter(Path(weights), imgsz=imgsz, threshold=threshold,
                                   encoder=encoder, device=device)

    def mask_volume(self, flair_path) -> np.ndarray:
        vol = normalize_mri_volume(nib.load(str(flair_path)).get_fdata())  # HWZ uint8
        self.last_gray = vol  # переиспользуется для анатомических признаков (авто-класс)
        H, W, Z = vol.shape
        out = np.zeros((H, W, Z), np.uint8)
        for z in range(Z):
            pz, nz = max(z - 1, 0), min(z + 1, Z - 1)
            rgb = np.stack([vol[:, :, pz], vol[:, :, z], vol[:, :, nz]], axis=2).astype(np.uint8)
            prob, w0, h0 = self.seg.probability_map(rgb)
            m = (prob > self.seg.threshold).astype(np.uint8)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            out[:, :, z] = m
        return out


if __name__ == "__main__":
    import time
    fp = sys.argv[1]
    t = time.time()
    d = Detector2p5d()
    m = d.mask_volume(fp)
    print(f"mask {m.shape} lesion-voxels={int(m.sum())} ({time.time()-t:.1f}s)")
