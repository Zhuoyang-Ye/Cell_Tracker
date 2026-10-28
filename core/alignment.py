from __future__ import annotations
import numpy as np
from scipy.optimize import fmin
from typing import Callable, Optional
from .roi_mask import create_roi_mask

# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _build_transform(v: np.ndarray, image_size: tuple):
    """Decompose v into shift, centre m, and combined transform matrix T."""
    alpha = v[0] / 360.0 * 2.0 * np.pi
    shift = np.array([v[1], v[2]], dtype=float)
    S = np.array([[v[3], v[5]], [v[5], v[4]]], dtype=float)
    Omega = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha),  np.cos(alpha)]], dtype=float)
    m = np.array([image_size[1] / 2.0, image_size[0] / 2.0])
    T = S @ Omega
    return shift, m, T


def _transform_binary_image(
    img_in: np.ndarray,
    v: np.ndarray,
    image_size: tuple,
) -> np.ndarray:
    """Apply v to a binary image (used only during optimisation)."""
    shift, m, T = _build_transform(v, image_size)
    h, w = image_size
    img_out = np.zeros((h, w), dtype=bool)

    rows, cols = np.nonzero(img_in)
    if len(rows) == 0:
        return img_out

    roi = np.stack([cols.astype(float), rows.astype(float)], axis=0)  # [x; y]
    r_new = np.ceil(m[:, None] + shift[:, None] + T @ (roi - m[:, None])).astype(int)

    valid = (r_new[0] >= 0) & (r_new[0] < w) & (r_new[1] >= 0) & (r_new[1] < h)
    img_out[r_new[1, valid], r_new[0, valid]] = True
    return img_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_alignment(
    xpix: list,
    ypix: list,
    v: np.ndarray,
    image_size: tuple,
) -> tuple[list, list]:
    """Apply a 6-parameter affine transform to ROI coordinate lists.

    Parameters
    ----------
    xpix, ypix : lists of int arrays  (0-indexed, suite2p convention)
    v : (6,) array  [angle_deg, shift_x, shift_y, scale_x, scale_y, shear]
    image_size : (H, W)

    Returns
    -------
    xpix_out, ypix_out : transformed lists (coordinates outside image dropped)
    """
    shift, m, T = _build_transform(v, image_size)
    h, w = image_size

    xpix_out, ypix_out = [], []
    for x, y in zip(xpix, ypix):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) == 0:
            xpix_out.append(np.array([], dtype=np.int32))
            ypix_out.append(np.array([], dtype=np.int32))
            continue

        roi = np.stack([x, y], axis=0)  # [x; y]
        r_new = np.ceil(m[:, None] + shift[:, None] + T @ (roi - m[:, None])).astype(int)

        valid = (r_new[0] >= 0) & (r_new[0] < w) & (r_new[1] >= 0) & (r_new[1] < h)
        xpix_out.append(r_new[0, valid].astype(np.int32))
        ypix_out.append(r_new[1, valid].astype(np.int32))

    return xpix_out, ypix_out


def find_alignment(
    xpix_a: list,
    ypix_a: list,
    xpix_b: list,
    ypix_b: list,
    image_size: tuple,
    param: dict,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Find the optimal 6-parameter affine transform aligning session A onto B.

    Uses Nelder-Mead optimisation with multiple restarts (mirrors find_alignment.m).

    Parameters
    ----------
    xpix_a, ypix_a : alignment-subset ROI coords for session A
    xpix_b, ypix_b : alignment-subset ROI coords for session B
    image_size : (H, W)
    param : dict with keys mask_radius, opt_scale, max_iter, tol_fun, tol_x,
                          v_lower, v_upper
    progress_callback : called as (k, total) after each restart

    Returns
    -------
    v_opt : (6,) array  [angle_deg, shift_x, shift_y, scale_x, scale_y, shear]
    """
    img_a = create_roi_mask(xpix_a, ypix_a, image_size, param.get('mask_radius', 0))
    img_b = create_roi_mask(xpix_b, ypix_b, image_size, param.get('mask_radius', 0))

    opt_scale = param.get('opt_scale', 0.5)
    stepsize = np.array([0.25, 10.0, 10.0, 0.025, 0.025, 0.025]) * opt_scale
    stepsize_machine = stepsize * 20.0  # fminsearch uses 5% of v0 as step size

    v0_base = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    v_lower = np.array(param.get('v_lower', [-15, -50, -50, 0.85, 0.85, -0.15]))
    v_upper = np.array(param.get('v_upper', [15, 50, 50, 1.15, 1.15, 0.15]))

    start_points = [
        v0_base + stepsize_machine,
        v0_base + stepsize_machine + np.array([2, 0, 0, 0, 0, 0.0]),
        v0_base + stepsize_machine + np.array([-2, 0, 0, 0, 0, 0.0]),
        v0_base + stepsize_machine + np.array([0, 5, 5, 0, 0, 0.0]),
        v0_base + stepsize_machine + np.array([0, -5, -5, 0, 0, 0.0]),
    ]

    def objective(v: np.ndarray) -> float:
        v_actual = v - stepsize_machine
        if np.any(v_actual < v_lower) or np.any(v_actual > v_upper):
            return 1e10
        img_t = _transform_binary_image(img_a, v_actual, image_size)
        return -float(np.count_nonzero(img_t & img_b))

    best_overlap = -np.inf
    best_v_raw = start_points[0].copy()
    failures = []

    for k, v0 in enumerate(start_points):
        try:
            v_raw = fmin(
                objective, v0,
                maxfun=param.get('max_iter', 2000),
                ftol=param.get('tol_fun', 1e-4),
                xtol=param.get('tol_x', 1e-4),
                disp=False,
            )
            overlap = -objective(v_raw)
            if overlap > best_overlap:
                best_overlap = overlap
                best_v_raw = v_raw.copy()
        except Exception as exc:
            failures.append(str(exc))
        if progress_callback:
            progress_callback(k + 1, len(start_points))

    if not np.isfinite(best_overlap) or best_overlap < 0:
        raise RuntimeError('Alignment failed for every valid start: ' + '; '.join(failures))
    return best_v_raw - stepsize_machine
