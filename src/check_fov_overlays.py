# src/check_fov_overlays.py
import os, random
from pathlib import Path
import cv2
import numpy as np

THRESH = 0.05  # 5% deviation threshold

def compute_offset(img_gray, mask_u8):
    """Return (dx, dy, cx, cy): normalized offsets & mask centroid in pixels."""
    h, w = img_gray.shape
    m = (mask_u8 > 0).astype(np.uint8)
    M = cv2.moments(m)
    if M["m00"] == 0:
        # empty mask (shouldn't happen), treat as centered
        return 0.0, 0.0, w / 2.0, h / 2.0
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    dx = abs(cx - w / 2) / w
    dy = abs(cy - h / 2) / h
    return float(dx), float(dy), float(cx), float(cy)

def overlay(img_gray, mask_u8, alpha=0.35, draw_centers=True):
    # 3-channel gray
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(rgb); red[..., 2] = 255
    m = (mask_u8 > 0)[..., None]
    out = (rgb * (1 - alpha) + red * alpha).astype(np.uint8)
    out = np.where(m, out, rgb)

    if draw_centers:
        h, w = img_gray.shape
        dx, dy, cx, cy = compute_offset(img_gray, mask_u8)
        # image center (green) & mask centroid (blue)
        cv2.circle(out, (w // 2, h // 2), 6, (0, 255, 0), -1)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 6, (255, 0, 0), -1)
        # optional: put small text with dx/dy
        txt = f"dx={dx:.3f}, dy={dy:.3f}"
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return out

def save_grid(pairs, out_path, pad=8, bg=255):
    """Make a simple 2-column grid: [processed, overlay] for sampled images."""
    if not pairs:
        return
    h = max(p[0].shape[0] for p in pairs)
    w = max(p[0].shape[1] for p in pairs)
    rows = []
    for proc, ov in pairs:
        ph = cv2.copyMakeBorder(proc, 0, h - proc.shape[0], 0, w - proc.shape[1],
                                cv2.BORDER_CONSTANT, value=bg)
        oh = cv2.copyMakeBorder(ov,   0, h - ov.shape[0],   0, w - ov.shape[1],
                                cv2.BORDER_CONSTANT, value=(bg, bg, bg))
        row = np.concatenate([cv2.cvtColor(ph, cv2.COLOR_GRAY2BGR), oh], axis=1)
        rows.append(row)
    canvas = rows[0]
    for r in rows[1:]:
        canvas = np.concatenate([canvas, r], axis=0)
    canvas = cv2.copyMakeBorder(canvas, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                value=(bg, bg, bg))
    cv2.imwrite(str(out_path), canvas)

def main(split="train", k=8, preproc_root="preprocessed", thresh=THRESH):
    img_dir = Path(preproc_root) / split / "images"
    fov_dir = Path(preproc_root) / split / "fov_masks"
    out_dir = Path("debug") / "fov_overlays" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    if not imgs:
        print(f"No images in {img_dir}")
        return

    sample = random.sample(imgs, min(k, len(imgs)))
    pairs = []
    outliers = []

    report_lines = []
    print(f"Sampling {len(sample)} images from {split}...")
    for p in sample:
        proc = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        mask_path = fov_dir / (p.stem + ".png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if proc is None or mask is None:
            continue

        dx, dy, cx, cy = compute_offset(proc, mask)
        flag = max(dx, dy) > thresh
        tag = "OUTLIER" if flag else ""
        print(f"{p.name}: dx={dx:.3f}, dy={dy:.3f} {tag}")
        report_lines.append(f"{p.name}, dx={dx:.4f}, dy={dy:.4f}, flag={tag}")

        ov = overlay(proc, mask, alpha=0.35, draw_centers=True)
        pairs.append((proc, ov))

        if flag:
            outliers.append(p.name)
            # save per-image overlay for the outlier
            ov_path = out_dir / f"{p.stem}_outlier_dx{dx:.3f}_dy{dy:.3f}.png"
            cv2.imwrite(str(ov_path), ov)

    # Save grid of the sampled images (processed vs overlay)
    grid_path = out_dir / f"{split}_overlays.png"
    save_grid(pairs, grid_path)
    print(f"Saved overlays grid to {grid_path}")

    # Write a small report (outliers + all dx/dy)
    report_path = out_dir / f"{split}_offset_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        if outliers:
            f.write("\n\nOUTLIERS (> {thr:.2%}):\n".format(thr=thresh))
            for n in outliers:
                f.write(n + "\n")
    print(f"Wrote report to {report_path}")
    if outliers:
        print(f"Flagged {len(outliers)} outlier(s) (> {thresh*100:.0f}%):", outliers)

if __name__ == "__main__":
    # change split to "val" or "test" if you want; change k for sample size
    main(split="train", k=8, preproc_root="preprocessed", thresh=THRESH)
