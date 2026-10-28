from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from .roi_mask import create_roi_mask


def _flat_pixels(xpix_roi, ypix_roi, h, w) -> np.ndarray:
    """Return sorted unique linear pixel indices for one ROI."""
    x = np.asarray(xpix_roi, dtype=np.int32)
    y = np.asarray(ypix_roi, dtype=np.int32)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return np.unique(y[valid].astype(np.int64) * w + x[valid].astype(np.int64))


def compute_iou_matrix(
    xpix_a: list,
    ypix_a: list,
    xpix_b: list,
    ypix_b: list,
    image_size: tuple,
    mask_radius: int = 0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Compute pairwise IOU matrix between two aligned ROI sets.

    Returns (n_a, n_b) float64 array with values in [0, 1].

    When mask_radius == 0 (default), uses fast flat-index intersection.
    When mask_radius > 0, falls back to binary mask images.
    """
    h, w = image_size
    n_a = len(xpix_a)
    n_b = len(xpix_b)
    iou_matrix = np.zeros((n_a, n_b), dtype=np.float64)
    if n_a == 0 or n_b == 0:
        return iou_matrix

    if mask_radius == 0:
        # Fast path: flat sorted pixel indices, no image allocation needed
        flat_b = [_flat_pixels(xpix_b[j], ypix_b[j], h, w) for j in range(n_b)]
        size_b = np.array([len(f) for f in flat_b], dtype=np.int64)

        for i in range(n_a):
            fa = _flat_pixels(xpix_a[i], ypix_a[i], h, w)
            size_ai = len(fa)
            for j in range(n_b):
                inter = len(np.intersect1d(fa, flat_b[j], assume_unique=True))
                union = size_ai + size_b[j] - inter
                if union > 0:
                    iou_matrix[i, j] = inter / union

            if progress_callback and (i % 20 == 0 or i == n_a - 1):
                progress_callback(i + 1, n_a)
    else:
        # Mask path: needed when dilation is applied
        masks_b = np.stack(
            [create_roi_mask(xpix_b, ypix_b, image_size, mask_radius, [j])
             for j in range(n_b)],
            axis=0,
        )  # (n_b, H, W) bool

        for i in range(n_a):
            mask_a = create_roi_mask(xpix_a, ypix_a, image_size, mask_radius, [i])
            inter = np.count_nonzero(mask_a[None] & masks_b, axis=(1, 2))
            union = np.count_nonzero(mask_a[None] | masks_b, axis=(1, 2))
            nz = union > 0
            iou_matrix[i, nz] = inter[nz] / union[nz]

            if progress_callback and (i % 20 == 0 or i == n_a - 1):
                progress_callback(i + 1, n_a)

    return iou_matrix
