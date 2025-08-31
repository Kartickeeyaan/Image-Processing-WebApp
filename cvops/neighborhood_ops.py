# cvops/neighborhood_ops.py
import cv2
import numpy as np
from .base import Operation

class MeanFilter(Operation):
    name = "Mean Filter"
    category = "Neighbourhood operations"

    def apply(self, img, **kwargs):
        ksize = int(kwargs.get("ksize", 3))
        if ksize < 1: ksize = 1
        if ksize % 2 == 0: ksize += 1
        return cv2.blur(img, (ksize, ksize))

class MedianFilter(Operation):
    name = "Median Filter"
    category = "Neighbourhood operations"

    def apply(self, img, **kwargs):
        ksize = int(kwargs.get("ksize", 3))
        if ksize < 3: ksize = 3
        if ksize % 2 == 0: ksize += 1
        return cv2.medianBlur(img, ksize)

class GaussianFilter(Operation):
    name = "Gaussian Filter"
    category = "Neighbourhood operations"

    def apply(self, img, **kwargs):
        ksize = int(kwargs.get("ksize", 5))
        if ksize < 1: ksize = 1
        if ksize % 2 == 0: ksize += 1
        sigma = float(kwargs.get("sigma", 1.0))
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

class SobelFilter(Operation):
    name = "Sobel Filter"
    category = "Neighbourhood operations"

    def apply(self, img, **kwargs):
        dx = int(kwargs.get("dx", 1))
        dy = int(kwargs.get("dy", 0))
        ksize = int(kwargs.get("ksize", 3))
        if ksize not in (1, 3, 5, 7):
            if ksize % 2 == 0:
                ksize += 1
            if ksize < 1:
                ksize = 3
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, dy, dx, ksize=ksize) if (dx==0 and dy==1) else None
        # magnitude: if both directions, combine; else use absolute
        if gy is not None:
            mag = cv2.magnitude(gx, gy)
        else:
            mag = np.abs(gx)
        mag = np.uint8(np.clip((mag / mag.max()) * 255.0, 0, 255)) if mag.max() > 0 else np.zeros_like(gray, dtype=np.uint8)
        return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
