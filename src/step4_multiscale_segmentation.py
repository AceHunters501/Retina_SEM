#!/usr/bin/env python


import argparse
from pathlib import Path

import cv2
import numpy as np


def group_sem_files_by_image(sem_root: Path):
    """
    sem_root/
        scale_*/split_*/ *_sem.npz  (e.g. train/val/test)

    Returns
    -------
    img_to_paths : dict[str, list[Path]]
        image_id -> list of sem paths (one per scale)
    """
    img_to_paths = {}

    # Go through each scale folder
    scale_dirs = [p for p in sem_root.iterdir() if p.is_dir()]
    scale_dirs.sort()

    for sdir in scale_dirs:
        # Look for *_sem.npz in ALL nested subfolders (train/val/test)
        for sem_path in sorted(sdir.rglob("*_sem.npz")):
            base = sem_path.stem          # e.g. FIVES_0001_sem
            image_id = base[:-4] if base.endswith("_sem") else base  # -> FIVES_0001
            img_to_paths.setdefault(image_id, []).append(sem_path)

    return img_to_paths



def load_gray_and_fov_from_sem(sem_path: Path):
    """
    Uses image_path and fov_path stored in *_sem.npz to load grayscale and FOV.
    """
    data = np.load(sem_path, allow_pickle=True)
    img_path = Path(str(data["image_path"]))
    fov_path = Path(str(data["fov_path"]))

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")

    fov_u8 = cv2.imread(str(fov_path), cv2.IMREAD_GRAYSCALE)
    if fov_u8 is None:
        raise FileNotFoundError(f"Could not read FOV at {fov_path}")

    return gray, fov_u8


def fuse_scales_for_image(
    sem_paths,
    thr: float = 0.78,
    intensity_thr: int | None = None,
    min_area: int | None = 80,
):
    """
    Core fusion + thresholding for a single image.

    Returns
    -------
    vessel_mask : np.ndarray uint8, shape (H,W)
        0 background, 255 vessel
    """
    # Load gray + FOV from first scale (same for all scales)
    gray_u8, fov_u8 = load_gray_and_fov_from_sem(sem_paths[0])
    H, W = gray_u8.shape

    prob_scales = []

    for sem_path in sem_paths:
        data = np.load(sem_path, allow_pickle=True)
        p = data["vessel_prob"]          # (K,)
        labels = data["labels"]          # (H,W)

        # Map node probs to pixels
        pmap = p[labels]

        # Restrict to FOV
        pmap = pmap * (fov_u8 > 0)

        prob_scales.append(pmap.astype(np.float32))

    # Simple average fusion across scales
    p_stack = np.stack(prob_scales, axis=0)  # (num_scales, H, W)
    p_final = p_stack.mean(axis=0)

    # Optional intensity suppression: down-weight very bright pixels
    if intensity_thr is not None:
        p_final = p_final * (gray_u8 < intensity_thr)

    p_final = np.clip(p_final, 0.0, 1.0)

    # Threshold
    binary = (p_final >= thr).astype(np.uint8)
    binary = binary * (fov_u8 > 0).astype(np.uint8)

    # Optional: remove tiny components
    if min_area is not None and min_area > 0:
        binary_u8 = (binary * 255).astype(np.uint8)
        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
            binary_u8, connectivity=8
        )

        areas = stats[:, cv2.CC_STAT_AREA]
        cleaned = np.zeros_like(binary, dtype=np.uint8)
        for lbl in range(1, num_labels):  # label 0 is background
            if areas[lbl] >= min_area:
                cleaned[labels_cc == lbl] = 1
        binary = cleaned

    # Final 0/255 mask
    vessel_mask = (binary * 255).astype(np.uint8)
    return vessel_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sem-root",
        type=str,
        required=True,
        help="Root dir containing scale_* folders with *_sem.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save final vessel masks (PNG).",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.78,
        help="Probability threshold for vessel mask (default 0.78).",
    )
    parser.add_argument(
        "--intensity-thr",
        type=int,
        default=180,
        help="Optional grayscale cutoff to suppress bright non-vessels "
             "(set None or <0 to disable).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=80,
        help="Minimum connected-component area to keep (in pixels). "
             "Set 0 to disable.",
    )
    args = parser.parse_args()

    sem_root = Path(args.sem_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.intensity_thr is not None and args.intensity_thr < 0:
        intensity_thr = None
    else:
        intensity_thr = args.intensity_thr

    img_to_paths = group_sem_files_by_image(sem_root)

    print(f"Found {len(img_to_paths)} images to process.")
    for image_id, sem_paths in sorted(img_to_paths.items()):
        print(f"  Processing {image_id} with {len(sem_paths)} scales...")
        vessel_mask = fuse_scales_for_image(
            sem_paths,
            thr=args.thr,
            intensity_thr=intensity_thr,
            min_area=args.min_area,
        )

        out_path = out_dir / f"{image_id}_vessel_mask_thr{args.thr:.2f}.png"
        cv2.imwrite(str(out_path), vessel_mask)

    print("Done.")


if __name__ == "__main__":
    main()