#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess FIVES fundus images:
- Green-channel extraction
- CLAHE (local contrast)
- Optional resize
- FOV mask from processed image (robust circle-from-background approach)
- Train --> (train,val) split created ONLY under preprocessed/
- Test stays intact (no split)

Run from project root:
  python src/preprocess.py --limit 5
"""

import argparse
from pathlib import Path
import random
import cv2
import numpy as np
from tqdm import tqdm
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops

import cv2, numpy as np

def _circle_from_distance_transform(fg_mask_u8, r_bounds=(0.33, 0.52), shrink_px=4):
    """
    Given a coarse foreground mask (retina ≈ 1, background ≈ 0),
    fit the maximal inscribed circle using a distance transform.
    """
    m = (fg_mask_u8 > 0).astype(np.uint8)
    if m.sum() == 0:
        return None  # nothing to do

    # Distance to the boundary *inside* the foreground
    # Use L2 for true Euclidean distances
    dist = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
    # Center = argmax distance, Radius = max distance
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    r = float(dist[cy, cx])

    h, w = m.shape
    r_min = r_bounds[0] * min(h, w)
    r_max = r_bounds[1] * min(h, w)
    r = int(np.clip(r, r_min, r_max))

    # Rasterize the circle
    Y, X = np.ogrid[:h, :w]
    circ = ((X - cx)**2 + (Y - cy)**2 <= r*r).astype(np.uint8) * 255

    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*shrink_px+1, 2*shrink_px+1))
        circ = cv2.erode(circ, k, iterations=1)

    return circ


# ---------- utils ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

# ---------- preprocessing ----------
def to_green_clahe(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    g = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(g)

def resize_if_needed(img, target):
    if target is None:
        return img
    if isinstance(target, int):
        return cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    if isinstance(target, (tuple, list)) and len(target) == 2:
        H, W = int(target[0]), int(target[1])
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img

# ---- FOV helpers ----
def _solid_circle_mask(h, w, cx, cy, r, shrink_px=4):
    Y, X = np.ogrid[:h, :w]
    m = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    mask = (m.astype(np.uint8) * 255)
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink_px + 1, 2 * shrink_px + 1))
        mask = cv2.erode(mask, k, iterations=1)
    return mask

def make_fov_mask(proc_img, shrink_px=4, bg_pct=8, min_bg_area_ratio=0.002):
    """
    Robust FOV from outer boundary (background-first strategy):
      A) Find *background* (dark vignette) via low-intensity percentile
      B) Use minEnclosingCircle of the *foreground* (not background) to get the fundus disk
      C) If that fails or looks weird, Hough circle fallback
    Returns uint8 mask {0,255}.
    """
    # ---- prep uint8 ----
    if proc_img.dtype != np.uint8:
        img = cv2.normalize(proc_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img = proc_img.copy()

    h, w = img.shape
    img_area = h * w

    # ---------- A) estimate background via percentile ----------
    p = np.percentile(img, bg_pct)           # e.g., 8th percentile
    bg = (img <= p).astype(np.uint8) * 255   # background white for convenience

    # Clean up background mask a bit (connect the ring)
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))

    # If almost no background found, switch to edge-based proxy
    bg_area_ratio = float(bg.sum() // 255) / img_area
    if bg_area_ratio < min_bg_area_ratio:
        blur = cv2.GaussianBlur(img, (0,0), 3)
        edges = cv2.Canny(blur, 40, 120)
        bg = (edges > 0).astype(np.uint8) * 255

# ---------- B) min-enclosing circle of *foreground* ----------
    fg = cv2.bitwise_not(bg)

    # Keep largest CC so we get one main blob
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        comp = (labels == idx).astype(np.uint8) * 255
    else:
        comp = (fg > 0).astype(np.uint8) * 255

    # >>> NEW: fit maximal inscribed circle from distance transform
    dt_circle = _circle_from_distance_transform(
        comp, r_bounds=(0.33, 0.52), shrink_px=shrink_px
    )
    if dt_circle is not None:
        return dt_circle


    # ---------- C) Hough fallback (on edges) ----------
    blur = cv2.GaussianBlur(img, (0,0), 3)
    edges = cv2.Canny(blur, 30, 100)
    r_min = int(0.35 * min(h, w))
    r_max = int(0.49 * min(h, w))
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=min(h, w)//2, param1=120, param2=30,
                               minRadius=r_min, maxRadius=r_max)
    if circles is not None:
        x, y, r = np.round(circles[0, 0]).astype(int)
        return _solid_circle_mask(h, w, x, y, r, shrink_px=shrink_px)

    # ---------- Last-resort: centered circle heuristic ----------
    r = int(0.44 * min(h, w))
    return _solid_circle_mask(h, w, w//2, h//2, r, shrink_px=shrink_px)

def save_image(path: Path, img):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)

# ---------- split & processing ----------
def split_train_val(train_imgs, val_frac, seed=42):
    rng = random.Random(seed)
    idx = list(range(len(train_imgs)))
    rng.shuffle(idx)
    n_val = int(round(len(idx) * val_frac))
    val_idx = set(idx[:n_val])
    train_keep = [p for i, p in enumerate(train_imgs) if i not in val_idx]
    val_take  = [p for i, p in enumerate(train_imgs) if i in val_idx]
    return train_keep, val_take

import math

def _solid_circle_mask(h, w, cx, cy, r, shrink_px=0):
    Y, X = np.ogrid[:h, :w]
    m = (X - cx)**2 + (Y - cy)**2 <= r*r
    mask = (m.astype(np.uint8) * 255)
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*shrink_px+1, 2*shrink_px+1))
        mask = cv2.erode(mask, k, iterations=1)
    return mask

def adjust_mask_area(mask_u8, target=0.70, tol=0.05, shrink_px=4, r_bounds=(0.33, 0.52)):
    """
    If mask area ratio is outside [target±tol], redraw a circle (same center)
    with a scaled radius so area ratio ≈ target.
    """
    h, w = mask_u8.shape
    cur_ratio = (mask_u8 > 0).sum() / float(h*w)
    if cur_ratio == 0.0 or abs(cur_ratio - target) <= tol:
        return mask_u8  # empty or already fine

    # center from moments (fallback to image center)
    m = cv2.moments((mask_u8 > 0).astype(np.uint8))
    cx = int(m["m10"]/m["m00"]) if m["m00"] != 0 else w//2
    cy = int(m["m01"]/m["m00"]) if m["m00"] != 0 else h//2

    # current equivalent radius from area
    r_eq = math.sqrt(((mask_u8 > 0).sum()) / math.pi)
    # scale to target area
    scale = math.sqrt(target / max(cur_ratio, 1e-6))
    r_new = r_eq * scale

    # clamp to reasonable bounds (as fraction of min side)
    r_min = r_bounds[0] * min(h, w)
    r_max = r_bounds[1] * min(h, w)
    r_new = int(np.clip(r_new, r_min, r_max))

    return _solid_circle_mask(h, w, cx, cy, r_new, shrink_px=shrink_px)


def process_set(img_paths,
                out_img_dir: Path,
                out_fov_dir: Path,
                target_size,
                clip_limit,
                tile_grid,
                show_progress=True,
                limit=-1):
    ensure_dir(out_img_dir)
    ensure_dir(out_fov_dir)

    if limit is not None and limit > 0:
        img_paths = img_paths[:limit]

    iterator = tqdm(img_paths, desc=f"Preprocessing -> {out_img_dir.parent.name}") if show_progress else img_paths
    for p in iterator:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        proc = to_green_clahe(bgr, clip_limit=clip_limit, tile_grid_size=tile_grid)
        proc = resize_if_needed(proc, target_size)
        fov  = make_fov_mask(proc)
        fov  = adjust_mask_area(fov, target=0.70, tol=0.05, shrink_px=4, r_bounds=(0.33, 0.52))

        out_img_path = out_img_dir / p.name
        out_fov_path = out_fov_dir / (p.stem + ".png")

        save_image(out_img_path, proc)
        save_image(out_fov_path, fov)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="dataset", help="Root with raw dataset/")
    ap.add_argument("--out_root",  type=str, default="preprocessed", help="Where to write processed data/")
    ap.add_argument("--val_frac",  type=float, default=0.2, help="Fraction of TRAIN (raw) to send to VAL (under preprocessed)")
    ap.add_argument("--img_size",  type=int, default=1024, help="Resize to square size; set None to skip")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--clip_limit", type=float, default=2.0)
    ap.add_argument("--tile_grid",  type=int, nargs=2, default=[8, 8])
    ap.add_argument("--limit", type=int, default=-1, help="Process only first N images per split (for quick tests). -1 = all")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    tile_grid = tuple(args.tile_grid)
    target    = args.img_size if args.img_size is not None else None

    # auto-create preprocessed root
    ensure_dir(out_root)

    train_src = data_root / "train" / "Original"
    test_src  = data_root / "test"  / "Original"

    train_dst_img = out_root / "train" / "images"
    train_dst_fov = out_root / "train" / "fov_masks"
    val_dst_img   = out_root / "val"   / "images"
    val_dst_fov   = out_root / "val"   / "fov_masks"
    test_dst_img  = out_root / "test"  / "images"
    test_dst_fov  = out_root / "test"  / "fov_masks"

    train_imgs_all = list_images(train_src)
    test_imgs_all  = list_images(test_src)

    if len(train_imgs_all) == 0:
        raise FileNotFoundError(f"No training images found in {train_src}")

    train_keep, val_take = split_train_val(train_imgs_all, args.val_frac, seed=args.seed)

    process_set(train_keep, train_dst_img, train_dst_fov, target, args.clip_limit, tile_grid, limit=args.limit)
    process_set(val_take,   val_dst_img,   val_dst_fov,   target, args.clip_limit, tile_grid, limit=args.limit)
    process_set(test_imgs_all, test_dst_img, test_dst_fov, target, args.clip_limit, tile_grid, limit=args.limit)

    print(f"[Done] Train processed: {len(train_keep[:args.limit if args.limit>0 else None])} "
          f"| Val processed: {len(val_take[:args.limit if args.limit>0 else None])} "
          f"| Test processed: {len(test_imgs_all[:args.limit if args.limit>0 else None])}")
    print(f"Outputs -> {out_root.resolve()}")

if __name__ == "__main__":
    main()
