# cvops/contrast_ops.py
import numpy as np
from .base import Operation

class ContrastStretch(Operation):
    name = "Contrast Streching"
    category = "Contrast Streching"

    def apply(self, img, **kwargs):
        out = np.zeros_like(img, dtype=np.uint8)
        for c in range(img.shape[2]):
            channel = img[:, :, c].astype(np.float32)
            mn = channel.min()
            mx = channel.max()
            if mx - mn > 1e-5:
                stretched = (channel - mn) / (mx - mn) * 255.0
            else:
                stretched = channel
            out[:, :, c] = np.clip(stretched, 0, 255).astype(np.uint8)
        return out
