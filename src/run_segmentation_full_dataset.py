#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Final graph-cut (or fallback) segmentation over the *preprocessed* dataset.
# Assumes you have already run preprocess.py so that you have:
# preprocessed/
#   train/{images,fov_masks}
#   val/{images,fov_masks}
#   test/{images,fov_masks}
#
# Outputs under: outputs_seg/
#   <split>/masks/*.png            (binary vessel mask)
#   <split>/overlays/*.png         (result overlay on grayscale)
#   <split>/seed_vis/*.png         (red=FG seeds, green=BG seeds)
#   <split>/debug_boundaries/*.png (SLIC boundaries preview)
#   summary_<timestamp>.csv
#
# Seeds: FG = top vesselness percentile; BG = low vesselness + FOV ring.
# Graph cut: uses PyMaxflow if installed; else percentile threshold fallback.

import argparse
import csv
import math
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from tqdm import tqdm

# Optional imports (handled lazily)
_have_skimage = True
_have_maxflow = True
try:
    from skimage.segmentation import slic, mark_boundaries, find_boundaries
    from skimage.filters import frangi
    from skimage.util import img_as_float
except Exception:
    _have_skimage = False

try:
    import maxflow  # PyMaxflow
except Exception:
    _have_maxflow = False


# -------------------- IO utils --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path, exts=('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def save_png(path: Path, img):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


# -------------------- Image helpers --------------------
def normalize_u8(img):
    """Map to uint8 [0,255]."""
    if img.dtype == np.uint8:
        return img
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    out = (np.clip((img - mn) / (mx - mn), 0, 1) * 255.0).astype(np.uint8)
    return out


def overlay_mask(gray_u8, mask_u8, alpha=0.35):
    """Overlay binary mask in red on gray; also draw centers for quick QA."""
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(rgb); red[..., 2] = 255
    m = (mask_u8 > 0)[..., None]
    out = (rgb * (1 - alpha) + red * alpha).astype(np.uint8)
    out = np.where(m, out, rgb)

    h, w = gray_u8.shape
    M = cv2.moments((mask_u8 > 0).astype(np.uint8))
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) ; cy = int(M["m01"] / M["m00"])
        cv2.circle(out, (w // 2, h // 2), 5, (0, 255, 0), -1)  # image center
        cv2.circle(out, (cx, cy), 5, (255, 0, 0), -1)          # mask centroid
    return out


def draw_seed_vis(gray_u8, fg_seed, bg_seed):
    """Visualize seeds: red=FG, green=BG."""
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    rg = rgb.copy()
    rg[fg_seed > 0] = (0, 0, 255)
    rg[bg_seed > 0] = (0, 255, 0)
    # blend with gray for context
    out = cv2.addWeighted(rgb, 0.6, rg, 0.4, 0)
    return out


# -------------------- Vesselness + SLIC --------------------
def compute_vesselness(gray_u8, fov_u8, frangi_beta=0.5, frangi_gamma=15.0):
    if not _have_skimage:
        # Fallback: simple DoG-like enhancement
        g = gray_u8.astype(np.float32) / 255.0
        blur1 = cv2.GaussianBlur(g, (0, 0), 1.0)
        blur2 = cv2.GaussianBlur(g, (0, 0), 2.5)
        vn = np.clip(blur1 - blur2, 0, 1)
        vn *= (fov_u8 > 0).astype(np.float32)
        return vn

    from skimage.util import img_as_float
    from skimage.filters import frangi
    g = img_as_float(gray_u8)
    vn = frangi(g, beta=frangi_beta, gamma=frangi_gamma)  # 0..1
    vn = np.nan_to_num(vn, nan=0.0, posinf=1.0, neginf=0.0)
    vn *= (fov_u8 > 0)
    return vn.astype(np.float32)


def build_superpixels(gray_u8, n_segments=1200, compactness=0.1, sigma=0):
    if not _have_skimage:
        # Crude fallback: each pixel is its own "superpixel"
        labels = np.arange(gray_u8.size, dtype=np.int32).reshape(gray_u8.shape)
        return labels, 1 + labels.max()

    img3 = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    from skimage.segmentation import slic
    labels = slic(img3, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=0, channel_axis=-1)
    k = int(labels.max()) + 1
    return labels.astype(np.int32), k


def adjacency_from_labels(labels):
    """Return adjacency list of superpixels from 4-neighborhood boundaries."""
    h, w = labels.shape
    adj = [set() for _ in range(labels.max() + 1)]
    # right/left neighbors
    L = labels[:, :-1]
    R = labels[:, 1:]
    m = L != R
    a = L[m]; b = R[m]
    for u, v in zip(a, b):
        adj[u].add(int(v)); adj[v].add(int(u))

    # up/down neighbors
    U = labels[:-1, :]
    D = labels[1:, :]
    m = U != D
    a = U[m]; b = D[m]
    for u, v in zip(a, b):
        adj[u].add(int(v)); adj[v].add(int(u))

    return [sorted(list(s)) for s in adj]


# -------------------- Seed selection --------------------
def select_seeds(vn, fov_u8, labels, fg_top_pct=0.15, bg_low_pct=0.15, ring_px=8):
    """
    FG seeds: top vesselness percentile within FOV.
    BG seeds: low vesselness + a narrow ring near FOV boundary.
    Returns uint8 masks.
    """
    in_fov = (fov_u8 > 0)
    vn_f = vn[in_fov]
    if vn_f.size == 0:
        return np.zeros_like(fov_u8), np.zeros_like(fov_u8)

    # Percentile thresholds
    hi = np.percentile(vn_f, 100 * (1 - fg_top_pct))
    lo = np.percentile(vn_f, 100 * (bg_low_pct))

    fg = ((vn >= hi) & in_fov).astype(np.uint8) * 255
    bg = ((vn <= lo) & in_fov).astype(np.uint8) * 255

    # Add a ring near FOV boundary to background seeds
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    inner = cv2.erode((fov_u8 > 0).astype(np.uint8) * 255, k, iterations=1)
    ring = cv2.subtract((fov_u8 > 0).astype(np.uint8) * 255, inner)
    bg = cv2.bitwise_or(bg, ring)

    return fg, bg


# -------------------- Graph cut on superpixels --------------------
def graph_cut_segment(gray_u8, vn, labels, adj, fg_seed, bg_seed, lam_unary=5.0, beta_pair=20.0):
    """
    Graph cut on superpixels with unary costs from vesselness and pairwise from boundary contrast.
    Returns binary mask (uint8 0/255).
    """
    H, W = gray_u8.shape
    K = labels.max() + 1

    # Precompute per-superpixel stats
    means_vn = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mask = (labels == k)
        if mask.any():
            means_vn[k] = float(vn[mask].mean())
        else:
            means_vn[k] = 0.0

    # Seed constraints
    sp_fg = np.zeros(K, dtype=bool)
    sp_bg = np.zeros(K, dtype=bool)
    for k in range(K):
        sp_mask = (labels == k)
        if np.any(fg_seed[sp_mask] > 0):
            sp_fg[k] = True
        if np.any(bg_seed[sp_mask] > 0):
            sp_bg[k] = True

    if not _have_maxflow:
        # Fallback: classify by threshold; honor hard seeds
        thr = np.percentile(means_vn, 85.0) if means_vn.size > 0 else 0.0
        lab = (means_vn >= thr)
        lab[sp_bg] = False
        lab[sp_fg] = True
        seg = lab[labels].astype(np.uint8) * 255
        seg[(vn <= 0) | (labels < 0)] = 0
        return seg

    import maxflow
    g = maxflow.Graph[float](K, K * 6)
    nodeids = g.add_nodes(K)

    # Unary costs: encourage FG for high vesselness
    eps = 1e-6
    for k in range(K):
        v = float(means_vn[k])
        v = max(0.0, min(1.0, v))
        c_fg = lam_unary * (1.0 - v)  # high vn -> low cost to FG
        c_bg = lam_unary * (v)        # low vn -> low cost to BG
        if sp_fg[k]:
            c_fg = 0.0; c_bg = 1e6
        if sp_bg[k]:
            c_fg = 1e6; c_bg = 0.0
        g.add_tedge(nodeids[k], c_fg + eps, c_bg + eps)

    # Pairwise costs: contrast-sensitive Potts over adjacency
    mean_gray = np.zeros(K, dtype=np.float32)
    for k in range(K):
        m = (labels == k)
        if m.any():
            mean_gray[k] = gray_u8[m].mean()

    for u in range(K):
        for v in adj[u]:
            if v <= u:
                continue
            diff = abs(float(mean_gray[u]) - float(mean_gray[v])) / 255.0
            w = beta_pair * math.exp(-3.0 * (diff ** 2))
            g.add_edge(nodeids[u], nodeids[v], w, w)

    g.maxflow()
    lab = np.array([g.get_segment(nodeids[k]) == 0 for k in range(K)], dtype=bool)  # True=source/FG

    seg = lab[labels].astype(np.uint8) * 255
    seg[(vn <= 0) | (labels < 0)] = 0
    return seg


# -------------------- Pipeline per-image --------------------
def process_one(img_path: Path, fov_path: Path, out_dirs, cfg):
    gray_u8 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    fov_u8  = cv2.imread(str(fov_path), cv2.IMREAD_GRAYSCALE)
    if gray_u8 is None or fov_u8 is None:
        return {"ok": False, "reason": "missing_image_or_fov"}

    vn = compute_vesselness(gray_u8, fov_u8, frangi_beta=cfg["frangi_beta"], frangi_gamma=cfg["frangi_gamma"])
    labels, K = build_superpixels(gray_u8, n_segments=cfg["n_segments"], compactness=cfg["compactness"], sigma=0)

    adj = adjacency_from_labels(labels)
    fg_seed, bg_seed = select_seeds(vn, fov_u8, labels, fg_top_pct=cfg["fg_top_pct"], bg_low_pct=cfg["bg_low_pct"], ring_px=cfg["ring_px"])
    seg = graph_cut_segment(gray_u8, vn, labels, adj, fg_seed, bg_seed, lam_unary=cfg["lam_unary"], beta_pair=cfg["beta_pair"])

    # Save artifacts
    mask_path = out_dirs["masks"] / (img_path.stem + ".png")
    save_png(mask_path, seg)

    overlay_path = out_dirs["overlays"] / (img_path.stem + ".png")
    save_png(overlay_path, overlay_mask(gray_u8, seg, alpha=0.35))

    seeds_vis_path = out_dirs["seed_vis"] / (img_path.stem + ".png")
    save_png(seeds_vis_path, draw_seed_vis(gray_u8, fg_seed, bg_seed))

    if _have_skimage:
        from skimage.segmentation import find_boundaries
        b = find_boundaries(labels, mode="outer").astype(np.uint8) * 255
        b_rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        b_rgb[b > 0] = (0, 0, 255)
        dbg_path = out_dirs["debug_boundaries"] / (img_path.stem + ".png")
        save_png(dbg_path, b_rgb)

    return {
        "ok": True,
        "K": int(K),
        "mean_vn": float(vn[vn > 0].mean()) if (vn > 0).any() else 0.0,
        "fg_seed_px": int((fg_seed > 0).sum()),
        "bg_seed_px": int((bg_seed > 0).sum()),
        "used_maxflow": bool(_have_maxflow),
    }


# -------------------- Main loop --------------------
def run_split(split_root: Path, out_root: Path, cfg, csv_writer):
    img_dir = split_root / "images"
    fov_dir = split_root / "fov_masks"
    out_dirs = {
        "masks": out_root / "masks",
        "overlays": out_root / "overlays",
        "seed_vis": out_root / "seed_vis",
        "debug_boundaries": out_root / "debug_boundaries",
    }
    for d in out_dirs.values():
        ensure_dir(d)

    imgs = list_images(img_dir)
    n_ok = 0
    for p in tqdm(imgs, desc=f"[{split_root.parent.name}/{split_root.name}]"):
        fov_p = fov_dir / (p.stem + ".png")
        res = process_one(p, fov_p, out_dirs, cfg)
        row = {
            "split": split_root.name,
            "image": p.name,
            "status": "ok" if res.get("ok") else f"fail:{res.get('reason','?')}",
            "K_superpixels": res.get("K", -1),
            "mean_vesselness": f"{res.get('mean_vn', 0.0):.5f}",
            "fg_seed_px": res.get("fg_seed_px", 0),
            "bg_seed_px": res.get("bg_seed_px", 0),
            "used_maxflow": res.get("used_maxflow", False),
        }
        csv_writer.writerow(row)
        if res.get("ok"):
            n_ok += 1
    return len(imgs), n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_root", type=str, default="preprocessed", help="Root produced by preprocess.py")
    ap.add_argument("--out_root", type=str, default="outputs_seg", help="Where to write results")
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])

    # Vesselness/SLIC/Graph params
    ap.add_argument("--frangi_beta", type=float, default=0.5)
    ap.add_argument("--frangi_gamma", type=float, default=15.0)
    ap.add_argument("--n_segments", type=int, default=1200)
    ap.add_argument("--compactness", type=float, default=0.1)
    ap.add_argument("--fg_top_pct", type=float, default=0.15)
    ap.add_argument("--bg_low_pct", type=float, default=0.15)
    ap.add_argument("--ring_px", type=int, default=8)
    ap.add_argument("--lam_unary", type=float, default=5.0)
    ap.add_argument("--beta_pair", type=float, default=20.0)
    args = ap.parse_args()

    pre_root = Path(args.pre_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_root / f"summary_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["split", "image", "status", "K_superpixels", "mean_vesselness",
                      "fg_seed_px", "bg_seed_px", "used_maxflow"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        total, ok_total = 0, 0
        for split in args.splits:
            split_root = pre_root / split
            split_out = out_root / split
            ensure_dir(split_out)
            n_all, n_ok = run_split(split_root, split_out, vars(args), w)
            total += n_all; ok_total += n_ok
            print(f"Done {split}: {n_ok}/{n_all} images processed successfully.")

    print(f"[FINISHED] {ok_total}/{total} images processed. Results in: {out_root.resolve()}")
    if not _have_skimage:
        print("NOTE: scikit-image not found. Used simplified vesselness & no SLIC; quality will be lower.", file=sys.stderr)
    if not _have_maxflow:
        print("NOTE: PyMaxflow not found. Used percentile threshold fallback instead of graph cut.", file=sys.stderr)


if __name__ == "__main__":
    main()
