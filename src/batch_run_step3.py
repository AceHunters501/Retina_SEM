#!/usr/bin/env python3
"""
Batch runner for Retina_SEM Step-3 (SLED) that executes your EXISTING notebook per image,
while redirecting all Step-3 outputs into `outputs_sled/` by default (configurable).

It injects a small parameter cell before execution to set:
  - SPLIT, SAMPLE_ID
  - OUT_PARENT (default: "outputs_sled")

So you don't need to modify your notebook to change output directories.

Usage (run from project root):
  python src/batch_run_step3.py --split all
  python src/batch_run_step3.py --split test --limit 25
  python src/batch_run_step3.py --split train --nb Retina_SEM_SLED_Debug.ipynb
  python src/batch_run_step3.py --split all --out-parent outputs_sled

Requirements:
  pip install nbclient nbformat
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple
import nbformat
from nbclient import NotebookClient, CellExecutionError
import traceback

DEFAULT_NB = "Retina_SEM_SLED_Debug.ipynb"

def find_ids(root: Path, split: str) -> List[str]:
    img_dir = root / "preprocessed" / split / "images"
    ids = [p.stem for p in sorted(img_dir.glob("*.png"))]
    return ids

def has_sp_map(root: Path, split: str, sample_id: str) -> bool:
    # Step-2 superpixel boundary (input for Step-3); we REQUIRE this to exist
    sp = root / "outputs_seg" / split / "debug_boundaries" / f"{sample_id}.png"
    return sp.exists()

def build_param_cell(split: str, sample_id: str, out_parent: str) -> str:
    # Injected cell to set parameters & environment variables in the notebook runtime
    return f"""
# === injected params (do not edit) ===
import os
SPLIT = {split!r}
SAMPLE_ID = {sample_id!r}
OUT_PARENT = {out_parent!r}
os.environ['SPLIT'] = SPLIT
os.environ['SAMPLE_ID'] = SAMPLE_ID
os.environ['OUT_PARENT'] = OUT_PARENT
print('[Injected]', 'SPLIT=', SPLIT, 'SAMPLE_ID=', SAMPLE_ID, 'OUT_PARENT=', OUT_PARENT)
"""

def execute_notebook(base_nb_path: Path, split: str, sample_id: str, out_parent: str, timeout: int = 1800) -> Tuple[bool, str]:
    nb = nbformat.read(base_nb_path, as_version=4)

    # Prepend parameter cell
    param_cell = nbformat.v4.new_code_cell(build_param_cell(split, sample_id, out_parent))
    nb.cells.insert(0, param_cell)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={'metadata': {'path': str(base_nb_path.parent)}}
    )

    try:
        client.execute()
        return True, "ok"
    except CellExecutionError as e:
        return False, f"CellExecutionError: {e}"
    except Exception as e:
        return False, f"Exception: {e}\n{traceback.format_exc()}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root (contains preprocessed/, outputs_seg/)")
    ap.add_argument("--split", type=str, default="all", choices=["train","test","all"], help="Which split to run")
    ap.add_argument("--nb", type=str, default=DEFAULT_NB, help="Notebook filename to execute per image")
    ap.add_argument("--out-parent", type=str, default="outputs_sled", help="Top-level folder for Step-3 outputs")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of images per split (0 means no limit)")
    ap.add_argument("--timeout", type=int, default=1800, help="Per-notebook execution timeout (seconds)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    nb_path = (root / args.nb) if not Path(args.nb).is_absolute() else Path(args.nb)
    if not nb_path.exists():
        print(f"[ERROR] Notebook not found: {nb_path}", file=sys.stderr)
        sys.exit(1)

    splits = ["train","test"] if args.split == "all" else [args.split]

    total = 0
    ok = 0
    skipped = 0
    missing_sp = 0
    failures = []

    for split in splits:
        ids = find_ids(root, split)
        if args.limit > 0:
            ids = ids[:args.limit]

        print(f"\n=== Running split={split} | {len(ids)} images (limit={args.limit}) | OUT_PARENT={args.out_parent} ===")
        for sid in ids:
            total += 1
            if not has_sp_map(root, split, sid):
                missing_sp += 1
                skipped += 1
                print(f"[SKIP no SP] {split}/{sid}")
                continue

            print(f"[RUN] {split}/{sid} ... ", end="", flush=True)
            success, msg = execute_notebook(nb_path, split, sid, out_parent=args.out_parent, timeout=args.timeout)
            if success:
                ok += 1
                print("done")
            else:
                failures.append((split, sid, msg))
                print("FAIL")

    print("\n=== Summary ===")
    print(f"root        : {root}")
    print(f"notebook    : {nb_path}")
    print(f"splits      : {', '.join(splits)}")
    print(f"OUT_PARENT  : {args.out_parent}")
    print(f"total imgs  : {total}")
    print(f"ok          : {ok}")
    print(f"skipped     : {skipped} (missing SP maps: {missing_sp})")
    print(f"failures    : {len(failures)}")
    for split, sid, msg in failures[:20]:
        print(f"  - {split}/{sid}: {msg}")

    if failures:
        sys.exit(2)

if __name__ == "__main__":
    main()
