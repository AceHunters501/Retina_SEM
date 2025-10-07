# src/fix_small_fov_masks.py
from pathlib import Path
import cv2, numpy as np, math

SPLITS = ["train", "test"]     # which splits to patch
PREPROC_ROOT = Path("preprocessed")
THRESH_SMALL = 0.55            # patch masks below this area ratio
TARGET = 0.70                  # aim here
TOL = 0.05                     # ± tolerance
SHRINK_PX = 4                  # keep same edge shrink as before
R_BOUNDS = (0.31, 0.54)        # allow a bit wider than before to catch tough cases

def solid_circle_mask(h, w, cx, cy, r, shrink_px=0):
    Y, X = np.ogrid[:h, :w]
    m = (X - cx)**2 + (Y - cy)**2 <= r*r
    mask = (m.astype(np.uint8) * 255)
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*shrink_px+1, 2*shrink_px+1))
        mask = cv2.erode(mask, k, iterations=1)
    return mask

def adjust_mask_area(mask_u8, target=0.70, tol=0.05, shrink_px=4, r_bounds=(0.33, 0.52)):
    h, w = mask_u8.shape
    cur_ratio = (mask_u8 > 0).sum() / float(h*w)
    if cur_ratio == 0.0 or abs(cur_ratio - target) <= tol:
        return mask_u8
    m = cv2.moments((mask_u8 > 0).astype(np.uint8))
    cx = int(m["m10"]/m["m00"]) if m["m00"] != 0 else w//2
    cy = int(m["m01"]/m["m00"]) if m["m00"] != 0 else h//2
    r_eq = math.sqrt(((mask_u8 > 0).sum()) / math.pi)
    scale = math.sqrt(target / max(cur_ratio, 1e-6))
    r_new = r_eq * scale
    r_min = r_bounds[0] * min(h, w)
    r_max = r_bounds[1] * min(h, w)
    r_new = int(np.clip(r_new, r_min, r_max))
    return solid_circle_mask(h, w, cx, cy, r_new, shrink_px=shrink_px)

def main():
    total = 0
    patched = 0
    for split in SPLITS:
        img_dir = PREPROC_ROOT / split / "images"
        msk_dir = PREPROC_ROOT / split / "fov_masks"
        paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff"}])
        for p in paths:
            mpath = msk_dir / (p.stem + ".png")
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if m is None: 
                continue
            h, w = m.shape
            ratio = (m > 0).sum() / float(h*w)
            total += 1
            if ratio < THRESH_SMALL:
                newm = adjust_mask_area(m, target=TARGET, tol=TOL, shrink_px=SHRINK_PX, r_bounds=R_BOUNDS)
                cv2.imwrite(str(mpath), newm)
                patched += 1
                print(f"[patched] {split}/{p.name}: {ratio:.3f} -> {(newm>0).sum()/(h*w):.3f}")
    print(f"Done. Patched {patched}/{total} images.")

if __name__ == "__main__":
    main()
