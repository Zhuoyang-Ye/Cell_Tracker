from __future__ import annotations
import numpy as np


def create_roi_mask(
    xpix: list,
    ypix: list,
    image_size: tuple,
    mask_radius: int = 0,
    roi_indices=None,
) -> np.ndarray:
    """Render ROI pixel coordinates into a binary (H, W) boolean image.

    Parameters
    ----------
    xpix, ypix : lists of int arrays
        Column / row pixel indices per ROI (0-indexed, suite2p convention).
    image_size : (H, W)
    mask_radius : dilation radius in pixels; 0 = no dilation.
    roi_indices : optional subset of ROI indices to render.
    """
    h, w = image_size
    img = np.zeros((h, w), dtype=bool)

    if roi_indices is None:
        roi_indices = range(len(xpix))

    for i in roi_indices:
        x = np.asarray(xpix[i], dtype=np.int32)
        y = np.asarray(ypix[i], dtype=np.int32)
        if len(x) == 0:
            continue
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        img[y[valid], x[valid]] = True

    if mask_radius > 0:
        from skimage.morphology import disk, binary_dilation
        img = binary_dilation(img, disk(mask_radius))

    return img
