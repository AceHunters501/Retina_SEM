import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm



def run_sem_optimization_stable(
    s,
    w_seed,
    edges,
    w_e,
    num_iters: int = 300,
    lr: float = 0.01,
    lam_seed: float = 2.0,
    lam_smooth: float = 3.0,
    lam_ent: float = 0.05,
):
    """
    Structural-entropy-inspired optimization on node probabilities x in [0,1].

    J(x) = lam_seed   * sum_i w_seed[i] * (x_i - s_i)^2
         + lam_smooth * sum_(i,j) w_e * (x_i - x_j)^2
         + lam_ent    * sum_i H(x_i),  H(x) = -x log x - (1-x) log(1-x)
    """
    K = s.shape[0]
    x = s.copy().astype(np.float32)
    eps = 1e-4

    u = edges[0].astype(np.int64)
    v = edges[1].astype(np.int64)
    w_e = w_e.astype(np.float32)

    for _ in range(num_iters):
        x = np.clip(x, eps, 1.0 - eps)
        grad = np.zeros_like(x, dtype=np.float32)

        # seed fidelity
        grad += 2.0 * lam_seed * w_seed * (x - s)

        # graph smoothness
        diff = x[u] - x[v]
        contrib = 2.0 * lam_smooth * w_e * diff
        np.add.at(grad, u, contrib)
        np.add.at(grad, v, -contrib)

        # entropy term: dH/dx = -log(x/(1-x))
        ent_grad = -np.log(x / (1.0 - x))
        grad += lam_ent * ent_grad

        # normalize gradient for stability
        g_max = float(np.max(np.abs(grad)))
        if g_max > 0:
            grad /= (g_max + 1e-6)

        x -= lr * grad

    return np.clip(x, 0.0, 1.0)




def build_edge_weights(features: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Build similarity weights for each edge from node features.

    features: (K, D)
    edges:    (2, E)
    """
    F = features.astype(np.float32)
    u = edges[0].astype(np.int64)
    v = edges[1].astype(np.int64)

    diff = F[u] - F[v]          # (E, D)
    dist2 = np.sum(diff ** 2, axis=1)

    sigma2 = np.median(dist2) + 1e-6
    w_e = np.exp(-dist2 / sigma2).astype(np.float32)
    return w_e


def build_seeds(node_seeds: np.ndarray):
    """
    Build seed prior s and seed weight w_seed from node_seeds in {-1,0,1}.
    """
    K = node_seeds.shape[0]
    s = np.full(K, 0.5, dtype=np.float32)
    s[node_seeds == 1] = 1.0
    s[node_seeds == 0] = 0.0

    w_seed = np.zeros(K, dtype=np.float32)
    w_seed[node_seeds == 1] = 1.0
    w_seed[node_seeds == 0] = 1.0
    return s, w_seed


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)




def process_one_graph(graph_path: Path, out_dir: Path, args, scale_name: str) -> dict:
    """
    Load one *_graph.npz, run SEM, and save *_sem.npz with vessel probabilities.
    """
    data = np.load(graph_path, allow_pickle=True)

    features   = data["features"]
    edges      = data["edges"]
    node_seeds = data["node_seeds"]
    centroids  = data["centroids"]
    labels     = data["labels"]

    image_path = str(data["image_path"]) if "image_path" in data.files else ""
    fov_path   = str(data["fov_path"])   if "fov_path" in data.files else ""

    K = features.shape[0]
    E = edges.shape[1]

    w_e = build_edge_weights(features, edges)
    s, w_seed = build_seeds(node_seeds)

    x_sem = run_sem_optimization_stable(
        s=s,
        w_seed=w_seed,
        edges=edges,
        w_e=w_e,
        num_iters=args.num_iters,
        lr=args.lr,
        lam_seed=args.lam_seed,
        lam_smooth=args.lam_smooth,
        lam_ent=args.lam_ent,
    )

    ensure_dir(out_dir)
    out_path = out_dir / (graph_path.stem.replace("_graph", "") + "_sem.npz")

    np.savez(
        out_path,
        vessel_prob=x_sem.astype(np.float32),
        node_seeds=node_seeds,
        centroids=centroids,
        edges=edges,
        edge_weights=w_e,
        features=features,
        labels=labels,
        image_path=image_path,
        fov_path=fov_path,
        scale_name=scale_name,
        sem_params=dict(
            num_iters=args.num_iters,
            lr=args.lr,
            lam_seed=args.lam_seed,
            lam_smooth=args.lam_smooth,
            lam_ent=args.lam_ent,
        ),
    )

    return {
        "ok": True,
        "K": int(K),
        "E": int(E),
        "fg_nodes_0_5": int((x_sem >= 0.5).sum()),
        "fg_nodes_0_6": int((x_sem >= 0.6).sum()),
    }


def run_for_split(split: str, graph_root: Path, out_root: Path, args):
    scale_name = args.scale_name
    split_dir = graph_root / scale_name / split   # <-- scale-specific folder

    if not split_dir.exists():
        print(f"[{split}] Graph folder not found: {split_dir}")
        return

    graph_files = sorted(split_dir.glob("*_graph.npz"))
    if not graph_files:
        print(f"[{split}] No *_graph.npz files in {split_dir}")
        return

    if args.limit is not None and args.limit > 0:
        graph_files = graph_files[: args.limit]

    print(f"[{split}] {len(graph_files)} graphs found for scale={scale_name}.")

    out_dir_split = out_root / scale_name / split  # <-- parallel SEM folder
    n_ok = 0

    for gp in tqdm(graph_files, desc=f"{split}", ncols=80):
        res = process_one_graph(gp, out_dir_split, args, scale_name)
        if res.get("ok"):
            n_ok += 1

    print(f"[{split}] Done. graphs={len(graph_files)} ok={n_ok} failed={len(graph_files) - n_ok}")




def main():
    ap = argparse.ArgumentParser(
        description="Step 3: Structural Entropy Minimization on superpixel graphs (per scale)."
    )

    ap.add_argument(
        "--graph_root",
        type=str,
        default="superPixel_graph",
        help="Root directory containing <scale_name>/<split>/*_graph.npz.",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="superPixel_graph_sem",
        help="Output root for SEM results (<scale_name>/<split>/*_sem.npz).",
    )
    ap.add_argument(
        "--scale_name",
        type=str,
        required=True,
        help="Scale subfolder name, e.g. scale_K3500_C6.",
    )
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process.",
    )

    # SEM hyperparameters
    ap.add_argument("--num_iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lam_seed", type=float, default=2.0)
    ap.add_argument("--lam_smooth", type=float, default=3.0)
    ap.add_argument("--lam_ent", type=float, default=0.05)

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of graphs per split.",
    )

    args = ap.parse_args()

    graph_root = Path(args.graph_root)
    out_root = Path(args.out_root)

    print(f"Running SEM for scale_name = {args.scale_name}")
    for split in args.splits:
        run_for_split(split, graph_root, out_root, args)


if __name__ == "__main__":
    main()
