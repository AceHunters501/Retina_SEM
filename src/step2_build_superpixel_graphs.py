import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

# We reuse the low-level primitives from your existing script
import helpers_superpixel as spx


def list_images(folder: Path):
    """Return sorted list of png/jpg images in a folder."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_graph_from_superpixels(
    gray_u8: np.ndarray,
    vn: np.ndarray,
    labels: np.ndarray,
    fg_seed_px: np.ndarray,
    bg_seed_px: np.ndarray,
    fg_frac_thresh: float = 0.05,
    bg_frac_thresh: float = 0.05,
):
    """
    Build node features, edge list, and node-level seeds from superpixels.

    - features: (K, 4) = [mean_intensity, mean_vesselness, cy_norm, cx_norm]
    - edges:    (2, E) undirected edge list with u < v
    - node_seeds: (K,) in {-1, 0, 1}  (-1 unknown, 0 BG, 1 FG)
    - centroids: (K, 2) in pixel coordinates [cy, cx]
    """
    H, W = gray_u8.shape
    K = int(labels.max()) + 1

    features = np.zeros((K, 4), dtype=np.float32)
    centroids = np.zeros((K, 2), dtype=np.float32)
    node_seeds = np.full(K, -1, dtype=np.int8)

    for k in range(K):
        mask = (labels == k)
        if not mask.any():
            continue

        ys, xs = np.where(mask)
        cy = ys.mean()
        cx = xs.mean()

        mean_intensity = float(gray_u8[mask].mean()) / 255.0
        mean_vn = float(vn[mask].mean()) if vn[mask].size > 0 else 0.0

        features[k, 0] = mean_intensity
        features[k, 1] = mean_vn
        features[k, 2] = cy / H
        features[k, 3] = cx / W

        centroids[k, 0] = cy
        centroids[k, 1] = cx

        # ----- node-level seeding from pixel-level seeds -----
        n_pix = mask.sum()
        fg_pix = int((fg_seed_px[mask] > 0).sum())
        bg_pix = int((bg_seed_px[mask] > 0).sum())

        fg_frac = fg_pix / n_pix
        bg_frac = bg_pix / n_pix

        if fg_frac >= fg_frac_thresh and fg_frac > bg_frac:
            node_seeds[k] = 1        # foreground node
        elif bg_frac >= bg_frac_thresh and bg_frac > fg_frac:
            node_seeds[k] = 0        # background node
        else:
            node_seeds[k] = -1       # unknown
        # -----------------------------------------------

    # ----- adjacency (reuse your helper) -----
    adj = spx.adjacency_from_labels(labels)

    edges_u, edges_v = [], []
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if v > u:
                edges_u.append(u)
                edges_v.append(v)

    edges = np.vstack(
        [np.array(edges_u, dtype=np.int32), np.array(edges_v, dtype=np.int32)]
    )

    return {
        "features": features,      # (K, 4)
        "edges": edges,            # (2, E)
        "node_seeds": node_seeds,  # (K,)
        "centroids": centroids,    # (K, 2)
    }


def process_one_image(
    img_path: Path,
    fov_path: Path,
    cfg: dict,
    out_dir_graphs: Path,
):
    """Process a single image → build & save its graph."""
    gray_u8 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    fov_u8 = cv2.imread(str(fov_path), cv2.IMREAD_GRAYSCALE)

    if gray_u8 is None or fov_u8 is None:
        return {"ok": False, "reason": "missing_image_or_fov"}

    # 1) vesselness
    vn = spx.compute_vesselness(
        gray_u8,
        fov_u8,
        frangi_beta=cfg["frangi_beta"],
        frangi_gamma=cfg["frangi_gamma"],
    )

    # 2) superpixels
    labels, K = spx.build_superpixels(
        gray_u8,
        n_segments=cfg["n_segments"],
        compactness=cfg["compactness"],
        sigma=0,
    )

    # 3) pixel-level seeds (FG/BG)
    fg_seed_px, bg_seed_px = spx.select_seeds(
        vn,
        fov_u8,
        labels,
        fg_top_pct=cfg["fg_top_pct"],
        bg_low_pct=cfg["bg_low_pct"],
        ring_px=cfg["ring_px"],
    )

    # 4) build node-level graph
    graph = build_graph_from_superpixels(
        gray_u8,
        vn,
        labels,
        fg_seed_px,
        bg_seed_px,
        fg_frac_thresh=cfg["fg_frac_thresh"],
        bg_frac_thresh=cfg["bg_frac_thresh"],
    )

    # 5) save .npz
    ensure_dir(out_dir_graphs)
    out_path = out_dir_graphs / f"{img_path.stem}_graph.npz"

    np.savez(
        out_path,
        features=graph["features"],
        edges=graph["edges"],
        node_seeds=graph["node_seeds"],
        centroids=graph["centroids"],
        labels=labels.astype(np.int32),
        image_path=str(img_path),
        fov_path=str(fov_path),
        n_segments=cfg["n_segments"],
        compactness=cfg["compactness"],
        frangi_beta=cfg["frangi_beta"],
        frangi_gamma=cfg["frangi_gamma"],
        fg_top_pct=cfg["fg_top_pct"],
        bg_low_pct=cfg["bg_low_pct"],
        ring_px=cfg["ring_px"],
        fg_frac_thresh=cfg["fg_frac_thresh"],
        bg_frac_thresh=cfg["bg_frac_thresh"],
        scale_name=cfg["scale_name"],
    )

    node_seeds = graph["node_seeds"]
    return {
        "ok": True,
        "K": int(K),
        "n_fg_nodes": int((node_seeds == 1).sum()),
        "n_bg_nodes": int((node_seeds == 0).sum()),
        "n_unknown_nodes": int((node_seeds == -1).sum()),
    }


def run_for_split(split: str, cfg: dict):
    pre_root = Path(cfg["pre_root"])
    img_dir = pre_root / split / "images"
    fov_dir = pre_root / split / "fov_masks"

    imgs = list_images(img_dir)
    if cfg["limit"] is not None and cfg["limit"] > 0:
        imgs = imgs[: cfg["limit"]]

    out_root = Path(cfg["out_root"])
    # <-- IMPORTANT: put graphs inside a scale-specific subfolder
    out_dir_graphs = out_root / cfg["scale_name"] / split

    if not imgs:
        print(f"[{split}] No images found in {img_dir}")
        return

    print(
        f"[{split}] {len(imgs)} images | scale={cfg['scale_name']} | "
        f"n_segments={cfg['n_segments']} | compactness={cfg['compactness']}"
    )

    n_ok = 0
    for img_path in tqdm(imgs, desc=f"{split}", ncols=80):
        fov_path = fov_dir / img_path.name
        res = process_one_image(img_path, fov_path, cfg, out_dir_graphs)
        if res.get("ok"):
            n_ok += 1

    print(f"[{split}] Done. images={len(imgs)} ok={n_ok} failed={len(imgs) - n_ok}")


def main():
    ap = argparse.ArgumentParser(
        description="Step 2: build superpixel graphs (single scale) and save as .npz files."
    )
    ap.add_argument(
        "--pre_root",
        type=str,
        default="preprocessed",
        help="Root produced by preprocess.py",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="superPixel_graph",
        help="Where to write graph .npz files",
    )
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process",
    )
    ap.add_argument(
        "--n_segments",
        type=int,
        default=1500,
        help="Target number of SLIC superpixels",
    )
    ap.add_argument(
        "--compactness",
        type=float,
        default=10.0,
        help="SLIC compactness parameter",
    )

    # Optional tag to distinguish multiple scales on disk
    ap.add_argument(
        "--scale_name",
        type=str,
        default=None,
        help="Name for this scale (subfolder under out_root). "
             "If not provided, a name like K1500_C10p0 is auto-generated.",
    )

    # Vesselness + pixel-level seeding parameters
    ap.add_argument("--frangi_beta", type=float, default=0.5)
    ap.add_argument("--frangi_gamma", type=float, default=15.0)
    ap.add_argument("--fg_top_pct", type=float, default=0.15)
    ap.add_argument("--bg_low_pct", type=float, default=0.15)
    ap.add_argument("--ring_px", type=int, default=8)

    # Node-level seeding thresholds
    ap.add_argument(
        "--fg_frac_thresh",
        type=float,
        default=0.05,
        help="Min fraction of FG pixels in a superpixel to mark it as FG node.",
    )
    ap.add_argument(
        "--bg_frac_thresh",
        type=float,
        default=0.05,
        help="Min fraction of BG pixels in a superpixel to mark it as BG node.",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images per split (for debugging).",
    )

    args = ap.parse_args()

    # If user didn't give a scale_name, build one from n_segments + compactness
    if args.scale_name is None:
        # replace '.' with 'p' to keep folder name filesystem-friendly
        comp_str = str(args.compactness).replace(".", "p")
        scale_name = f"K{args.n_segments}_C{comp_str}"
    else:
        scale_name = args.scale_name

    cfg = {
        "pre_root": args.pre_root,
        "out_root": args.out_root,
        "n_segments": args.n_segments,
        "compactness": args.compactness,
        "frangi_beta": args.frangi_beta,
        "frangi_gamma": args.frangi_gamma,
        "fg_top_pct": args.fg_top_pct,
        "bg_low_pct": args.bg_low_pct,
        "ring_px": args.ring_px,
        "fg_frac_thresh": args.fg_frac_thresh,
        "bg_frac_thresh": args.bg_frac_thresh,
        "limit": args.limit,
        "scale_name": scale_name,
    }

    print(f"Using scale_name = {scale_name}")
    for split in args.splits:
        run_for_split(split, cfg)


if __name__ == "__main__":
    main()
