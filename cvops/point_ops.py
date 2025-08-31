# cvops/point_ops.py
import numpy as np
import cv2
from .base import Operation

class Negative(Operation):
    name = "Negative"
    category = "Point Operations"

    def apply(self, img, **kwargs):
        return 255 - img

class Gamma(Operation):
    name = "Gamma"
    category = "Point Operations"

    def apply(self, img, **kwargs):
        # param name = gamma (default 1.0)
        gamma = float(kwargs.get("gamma", 1.0))
        if gamma <= 0:
            gamma = 1.0
        img_norm = img.astype(np.float32) / 255.0
        out = np.power(img_norm, 1.0 / gamma)  # using 1/gamma gives brighter/darker as user expects often
        return (out * 255).clip(0, 255).astype(np.uint8)

class Log(Operation):
    name = "Log"
    category = "Point Operations"

    def apply(self, img, **kwargs):
        # param name = c (scaling constant)
        c = float(kwargs.get("c", 1.0))
        img_f = img.astype(np.float32)
        out = c * np.log1p(img_f)  # log(1 + I)
        # normalize to 0-255
        out = out - out.min()
        if out.max() > 0:
            out = out / out.max() * 255.0
        return out.clip(0, 255).astype(np.uint8)

class Threshold(Operation):
    name = "Threshold"
    category = "Point Operations"

    def apply(self, img, **kwargs):
        # param name = thresh (0-255)
        thresh = int(kwargs.get("thresh", 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
