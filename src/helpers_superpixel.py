"""
superpixel_graph_construction.py

Clean utility module for Retina-SEM Step 2.

Provides small, reusable building blocks:

- compute_vesselness(gray_u8, fov_u8, frangi_beta=0.5, frangi_gamma=15.0)
- build_superpixels(gray_u8, n_segments=1200, compactness=0.1, sigma=0)
- adjacency_from_labels(labels)
- select_seeds(vn, fov_u8, labels, fg_top_pct=0.15, bg_low_pct=0.15, ring_px=8)

These are used by:
    - notebooks/superpixel_graph_demo.ipynb
    - src/step2_build_superpixel_graphs.py

Nothing in this file performs segmentation or writes masks.
"""

from pathlib import Path
from typing import List

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Optional skimage dependency
# ---------------------------------------------------------------------------

_have_skimage = True
try:
    from skimage.filters import frangi
    from skimage.segmentation import slic
    from skimage.util import img_as_float
except Exception:  # pragma: no cover - fallback path
    _have_skimage = False


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
    """Return sorted list of image paths in a folder."""
    if not folder.exists():
        return []
    exts = {e.lower() for e in exts}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


# ---------------------------------------------------------------------------
# Core Step-2 helpers
# ---------------------------------------------------------------------------

def compute_vesselness(
    gray_u8: np.ndarray,
    fov_u8: np.ndarray,
    frangi_beta: float = 0.5,
    frangi_gamma: float = 15.0,
) -> np.ndarray:
    """
    Compute a vesselness map inside the FOV.

    Parameters
    ----------
    gray_u8 : (H, W) uint8
        Preprocessed grayscale fundus image.
    fov_u8 : (H, W) uint8
        FOV mask (non-zero inside FOV).
    frangi_beta, frangi_gamma : float
        Parameters for Frangi filter (if skimage is available).

    Returns
    -------
    vn : (H, W) float32
        Vesselness map in [0, 1], zero outside FOV.
    """
    in_fov = (fov_u8 > 0)

    if not _have_skimage:
        # Simple Difference-of-Gaussians fallback if skimage is unavailable.
        g = gray_u8.astype(np.float32) / 255.0
        blur1 = cv2.GaussianBlur(g, (0, 0), 1.0)
        blur2 = cv2.GaussianBlur(g, (0, 0), 2.5)
        vn = np.clip(blur1 - blur2, 0, 1)
        vn *= in_fov.astype(np.float32)
        return vn.astype(np.float32)

    # skimage-based Frangi
    g = img_as_float(gray_u8)
    vn = frangi(g, beta=frangi_beta, gamma=frangi_gamma)  # 0..1
    vn = np.nan_to_num(vn, nan=0.0, posinf=1.0, neginf=0.0)
    vn *= in_fov
    return vn.astype(np.float32)


def build_superpixels(
    gray_u8: np.ndarray,
    n_segments: int = 1200,
    compactness: float = 0.1,
    sigma: float = 0,
):
    """
    Run SLIC superpixel segmentation on a grayscale image.

    Parameters
    ----------
    gray_u8 : (H, W) uint8
    n_segments : int
        Target number of superpixels.
    compactness : float
        SLIC compactness parameter.
    sigma : float
        Gaussian smoothing prior to SLIC.

    Returns
    -------
    labels : (H, W) int32
        Superpixel labels in [0, K-1].
    K : int
        Number of superpixels.
    """
    if not _have_skimage:
        # Fallback: each pixel is its own superpixel
        labels = np.arange(gray_u8.size, dtype=np.int32).reshape(gray_u8.shape)
        return labels, int(labels.max()) + 1

    img3 = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    labels = slic(
        img3,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
    )
    labels = labels.astype(np.int32)
    K = int(labels.max()) + 1
    return labels, K


def adjacency_from_labels(labels: np.ndarray) -> List[set]:
    """
    Build a planar adjacency list between superpixels.

    Parameters
    ----------
    labels : (H, W) int
        Superpixel label map.

    Returns
    -------
    adj : list[set[int]]
        adj[k] is the set of neighbors of node k.
    """
    h, w = labels.shape
    K = int(labels.max()) + 1
    adj = [set() for _ in range(K)]

    # Horizontal neighbors
    L = labels[:, :-1]
    R = labels[:, 1:]
    m = L != R
    a = L[m]
    b = R[m]
    for u, v in zip(a, b):
        u = int(u)
        v = int(v)
        adj[u].add(v)
        adj[v].add(u)

    # Vertical neighbors
    U = labels[:-1, :]
    D = labels[1:, :]
    m = U != D
    a = U[m]
    b = D[m]
    for u, v in zip(a, b):
        u = int(u)
        v = int(v)
        adj[u].add(v)
        adj[v].add(u)

    # Convert sets to sorted lists for stable behavior
    return [sorted(list(s)) for s in adj]


def select_seeds(
    vn: np.ndarray,
    fov_u8: np.ndarray,
    labels: np.ndarray,
    fg_top_pct: float = 0.15,
    bg_low_pct: float = 0.15,
    ring_px: int = 8,
):
    """
    Select pixel-level foreground/background seeds from vesselness.

    Parameters
    ----------
    vn : (H, W) float32
        Vesselness map.
    fov_u8 : (H, W) uint8
        FOV mask (non-zero inside FOV).
    labels : (H, W) int
        Superpixel labels (unused directly, but kept for compatibility).
    fg_top_pct : float
        Fraction of highest vesselness pixels (inside FOV) to mark as FG seeds.
    bg_low_pct : float
        Fraction of lowest vesselness pixels (inside FOV) to mark as BG seeds.
    ring_px : int
        Thickness (in pixels) of additional BG ring at the FOV border.

    Returns
    -------
    fg_seed : (H, W) uint8
        Foreground seed mask (255 at seed pixels, 0 elsewhere).
    bg_seed : (H, W) uint8
        Background seed mask (255 at seed pixels, 0 elsewhere).
    """
    in_fov = fov_u8 > 0
    vn_f = vn[in_fov]

    if vn_f.size == 0:
        # No FOV or vesselness → no seeds
        return np.zeros_like(fov_u8, dtype=np.uint8), np.zeros_like(
            fov_u8, dtype=np.uint8
        )

    hi = np.percentile(vn_f, 100.0 * (1.0 - fg_top_pct))
    lo = np.percentile(vn_f, 100.0 * bg_low_pct)

    fg = ((vn >= hi) & in_fov).astype(np.uint8) * 255
    bg = ((vn <= lo) & in_fov).astype(np.uint8) * 255

    # Add a BG ring near FOV boundary to enforce background there
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1)
    )
    inner = cv2.erode((fov_u8 > 0).astype(np.uint8) * 255, k, iterations=1)
    ring = cv2.subtract((fov_u8 > 0).astype(np.uint8) * 255, inner)
    bg = cv2.bitwise_or(bg, ring)

    return fg, bg