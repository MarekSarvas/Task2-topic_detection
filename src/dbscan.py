import argparse
from pathlib import Path


import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Parse data for hyper param search for dbscan.")

    parser.add_argument(
        "--embedd_dir",
        type=str,
        default="data/callcenter_multiwoz",
        help="Path to directory containing embeddings"
    )
    parser.add_argument(
        "--elbow_dir",
        type=str,
        default="eps_elbows",
        help="Path to directory containing plots of distances between embeddings for different dim reduction sizes."
    )
    parser.add_argument(
        "--embedding_sizes", 
        nargs="+",
        type=int,
        help="A list of embedding sizes after dim reduction."
    )
    return parser.parse_args()


def cluster_dbscan(embeddings: np.ndarray, eps: float, min_samples: int) -> DBSCAN:
    dbscan_model = DBSCAN(eps=eps,
                          min_samples=min_samples,
                          metric="cosine")
    clusters = dbscan_model.fit(embeddings)
    return clusters


def print_elbow(distances: np.ndarray, fig_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(distances)

    ax.set_xlabel("data point", fontsize=12)
    ax.set_ylabel("distance", fontsize=12)
    ax.grid(True)  # Add grid lines
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')


def main(args):
    # check for the output directory
    elbow_dir = Path(args.elbow_dir)
    if not elbow_dir.exists():
        elbow_dir.mkdir(parents=True, exist_ok=True)

    for embedd_size in args.embedding_sizes:
        embeddings = np.load(Path(args.embedd_dir) / f"embeddings_{embedd_size}.npy")

        # perform KNN to compute embedding distances
        neighbors = NearestNeighbors(n_neighbors=2*embedd_size, metric="cosine")
        neighbors_fit = neighbors.fit(embeddings)
        distances, _ = neighbors_fit.kneighbors(embeddings)

        distances = np.sort(distances[:, (2*embedd_size)-1], axis=0)

        fig_path = elbow_dir / f"embeddings_{embedd_size}.png"
        print_elbow(distances, fig_path)

        # compute eps as mean distance from last neighbor
        # (should correspond to the elbow number)
        eps = np.mean(distances)
        clusters = cluster_dbscan(embeddings, eps, 2*embedd_size)

        test_embeddings = embeddings[clusters.labels_ != -1]
        clusters.labels_ = clusters.labels_[clusters.labels_ != -1]
        score = silhouette_score(test_embeddings, clusters.labels_)
        print(f"Mean: embedding_size={embedd_size}, silhouette_score={score}, eps={eps}")

        best_score = 0
        eps_best = 0
        for eps in distances[1000:]:
            clusters = cluster_dbscan(embeddings, eps, 2*embedd_size)
            # remove outliers
            test_embeddings = embeddings[clusters.labels_ != -1]
            clusters.labels_ = clusters.labels_[clusters.labels_ != -1]
            if len(set(clusters.labels_)) < 2:
                continue
            score = silhouette_score(test_embeddings, clusters.labels_)
            if score > best_score:
                best_score = score
                eps_best = eps
        print(f"Best: embedding_size={embedd_size}, score={best_score}, eps={eps_best}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
