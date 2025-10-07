# src/check_fov_stats.py
from pathlib import Path
import cv2
import numpy as np

def scan(split, preproc_root="preprocessed", too_small=0.55, too_large=0.90):
    img_dir = Path(preproc_root)/split/"images"
    msk_dir = Path(preproc_root)/split/"fov_masks"
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png",".jpg",".tif",".tiff",".jpeg"}])
    if not imgs:
        print(f"No images in {img_dir}")
        return
    ratios = []
    small, large = [], []
    for p in imgs:
        mask = cv2.imread(str(msk_dir/(p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if mask is None: 
            continue
        h, w = mask.shape
        r = (mask > 0).sum() / float(h*w)
        ratios.append(r)
        if r < too_small: small.append((p.name, r))
        if r > too_large: large.append((p.name, r))
    ratios = np.array(ratios)
    print(f"[{split}] n={len(ratios)}  mean={ratios.mean():.3f}  std={ratios.std():.3f}  "
          f"min={ratios.min():.3f}  max={ratios.max():.3f}")
    print(f"  too-small (<{too_small}): {len(small)}  | too-large (>{too_large}): {len(large)}")
    if small: print("  examples small:", small[:5])
    if large: print("  examples large:", large[:5])

if __name__ == "__main__":
    for split in ["train","val","test"]:
        scan(split)
