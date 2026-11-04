"""Data augmentation for training the custom CNN on aligned faces (train split only).

Works on aligned BGR uint8 faces (as saved by FaceService.align_face) and returns
the same format, so FaceService.preprocess_face is applied afterwards exactly as
in the app. The transforms keep the face aligned and the identity intact.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np


def build_train_augmentation() -> A.Compose:
    """Augmentation pipeline for aligned 112x112 faces in RGB order."""
    return A.Compose([
        # Faces are close to symmetric: a mirrored face is still the same person
        A.HorizontalFlip(p=0.5),
        # Small residual misalignment, as left by the 5-point alignment
        A.Affine(
            rotate=(-10, 10), translate_percent=(-0.05, 0.05), scale=(0.95, 1.05),
            border_mode=cv2.BORDER_REFLECT_101, p=0.7,
        ),
        # Lighting and color changes between photos of the same person
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=0, p=0.3),
        A.ToGray(p=0.1),
        # Lower quality: smaller and compressed faces, blur and sensor noise
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        A.ImageCompression(quality_range=(40, 90), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.1),
        # Partial occlusions (glasses, hands, microphones)
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.15, 0.3), hole_width_range=(0.15, 0.3), fill=0, p=0.3),
    ])


def augment_face(face_bgr: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Apply the transform to an aligned BGR uint8 face and return it in BGR uint8."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(transform(image=rgb)["image"], cv2.COLOR_RGB2BGR)
