"""2.5D U-Net lesion segmentation detector (alternative to the YOLO detector).

Box detection with an IoU criterion is unreliable for 5-15px MS lesions; pixel
segmentation with Dice/Tversky loss finds a lesion whenever any pixels overlap.
This subpackage trains a compact 2.5D U-Net on the reconstructed prev/cur/next
slices and turns its masks into lesion candidates for the anatomy classifier.
"""
