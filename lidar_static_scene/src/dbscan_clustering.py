"""
dbscan_clustering.py
--------------------
Implements Section 2.3 and 2.3.1 of the paper EXACTLY.

Paper algorithm per element (c, m):
─────────────────────────────────────────────────────────────
1. Set MinPts = max(100, 0.01 × total_points_in_element)
2. Set eps range: eps_initial=0.08, eps_max=0.40, step=0.01
3. Sweep eps:
     a. Apply DBSCAN(eps, MinPts) to distance list
     b. Count clusters (excluding outlier label -1)
     c. If MULTIPLE clusters:
           → compute silhouette score
           → track eps with MAX silhouette score
     d. If SINGLE cluster:
           → compute intra-cluster distance
           → track eps with MIN intra-cluster distance
4. Select best_eps:
     - If any eps produced multiple clusters:
         → best_eps = eps with highest silhouette score
     - Else (all single-cluster):
         → best_eps = eps with lowest intra-cluster distance
5. Run final DBSCAN with best_eps
6. Return: cluster labels, static_cluster_id, outlier mask
─────────────────────────────────────────────────────────────

Paper quote on selection criteria:
  "To select the best parameter, maximum silhouette score is
   selected in case of multiple cluster and minimum intra-cluster
   point distance is selected in case of single cluster."

Paper quote on MinPts:
  "minimum sample points are calculated subject to 1% of total
   points or minimum of 100 points to form a cluster"

Paper quote on eps range:
  "Minimum eps are selected 0.08 and maximum is 0.4. Methods
   evaluate the cluster at each eps value with the interval of 0.01"

Paper quote on beam divergence (Equation 3):
  "Arc Length (S) = D × θ"
  "The permissible deviation for laser hitting at the prescribed
   arc length is 3.49 mm/m."
  This sets the physical lower bound for eps (not the swept range).
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Result container                                                    #
# ------------------------------------------------------------------ #

@dataclass
class ElementClusterResult:
    """Stores clustering result for a single (channel, azimuth_bin) element."""
    channel:    int
    azimuth_bin: int
    distances:  np.ndarray          # all valid distances for this element

    # DBSCAN outcome
    labels:         np.ndarray      # per-distance DBSCAN label (-1=outlier)
    best_eps:       float
    best_min_pts:   int
    n_clusters:     int             # number of clusters (excl. outliers)
    had_multi_cluster: bool         # True if any eps produced >1 cluster

    # Static cluster
    static_cluster_id:  int         # which cluster is the static background
    static_distances:   np.ndarray  # distances belonging to static cluster
    static_representative: float    # single representative distance (median)

    # Outlier mask
    is_outlier: np.ndarray          # bool, True = NOT part of static scene

    # Quality metrics
    silhouette: Optional[float] = None
    intra_dist: Optional[float] = None


# ------------------------------------------------------------------ #
#  Intra-cluster distance                                              #
# ------------------------------------------------------------------ #

def intra_cluster_distance(distances: np.ndarray) -> float:
    """
    Mean pairwise absolute distance between all points in a cluster.
    For 1D data this is simply std-dev × sqrt(2/π) ≈ mean abs deviation.
    Paper uses "intra distance between points in cluster" as compactness.
    We implement as mean absolute deviation from cluster mean (MAD).
    """
    if len(distances) < 2:
        return 0.0
    mean = distances.mean()
    return float(np.mean(np.abs(distances - mean)))


# ------------------------------------------------------------------ #
#  Per-element DBSCAN with paper parameter selection                   #
# ------------------------------------------------------------------ #

def cluster_element(distances: np.ndarray,
                    channel: int,
                    azimuth_bin: int,
                    eps_initial: float = 0.08,
                    eps_max: float = 0.40,
                    eps_step: float = 0.01,
                    min_pts_fraction: float = 0.01,
                    min_pts_floor: int = 100,
                    representative: str = 'median') -> Optional[ElementClusterResult]:
    """
    Apply the full paper algorithm to a single (channel, azimuth_bin) element.

    Parameters
    ----------
    distances       : 1D array of aggregated distances across frames
    channel         : laser channel index (for logging/output)
    azimuth_bin     : azimuth bin index (for logging/output)
    eps_initial     : minimum eps to sweep (paper: 0.08)
    eps_max         : maximum eps to sweep (paper: 0.40)
    eps_step        : eps sweep interval  (paper: 0.01)
    min_pts_fraction: MinPts as fraction of total distances (paper: 0.01)
    min_pts_floor   : floor value for MinPts (paper: 100)
    representative  : 'median' or 'mean' for static distance summary

    Returns
    -------
    ElementClusterResult, or None if insufficient data.
    """
    n = len(distances)
    if n < 2:
        return None  # Not enough returns to cluster

    # ── Paper Section 2.3.1: MinPts ──────────────────────────────────
    min_pts = max(min_pts_floor, int(min_pts_fraction * n))
    min_pts = min(min_pts, n - 1)    # can't exceed data size

    # DBSCAN expects 2D input; our data is 1D distances
    X = distances.reshape(-1, 1)

    # ── Sweep eps as described in paper ──────────────────────────────
    eps_values = np.arange(eps_initial, eps_max + eps_step * 0.5, eps_step)
    eps_values = np.round(eps_values, 4)

    best_eps        = eps_initial
    best_score      = -np.inf      # for silhouette (higher = better)
    best_intra      = np.inf       # for intra-dist (lower = better)
    best_labels     = None
    had_multi       = False

    records_multi  = []   # (eps, silhouette_score, labels)
    records_single = []   # (eps, intra_dist, labels)

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_pts, algorithm='ball_tree')
        labels = db.fit_predict(X)

        unique_clusters = set(labels) - {-1}
        n_clusters = len(unique_clusters)

        if n_clusters == 0:
            # All noise — skip
            continue

        elif n_clusters > 1:
            # Multiple clusters: use silhouette score
            # Silhouette requires at least 2 labels that are non-noise
            non_noise_mask = labels != -1
            if non_noise_mask.sum() < 2 or len(set(labels[non_noise_mask])) < 2:
                continue
            try:
                sil = silhouette_score(X[non_noise_mask],
                                       labels[non_noise_mask])
            except Exception:
                continue
            records_multi.append((eps, sil, labels.copy()))
            had_multi = True

        else:
            # Single cluster
            cluster_mask = labels == list(unique_clusters)[0]
            intra = intra_cluster_distance(distances[cluster_mask])
            records_single.append((eps, intra, labels.copy()))

    # ── Select best eps ───────────────────────────────────────────────
    # Paper: "if all iterations have multiple clusters, the loop through
    #         eps_values are repeated, and the best eps is updated based
    #         on silhouette scores"
    # Paper: multi → max silhouette; single → min intra distance

    if had_multi and records_multi:
        # Pick max silhouette across multi-cluster records
        records_multi.sort(key=lambda x: x[1], reverse=True)
        best_eps, best_sil, best_labels = records_multi[0]
        best_silhouette = best_sil
        best_intra_val  = None
    elif records_single:
        # All were single cluster — pick min intra distance
        records_single.sort(key=lambda x: x[1])
        best_eps, best_intra_val, best_labels = records_single[0]
        best_silhouette = None
    elif records_multi:
        # Had multi but no single — still use best silhouette
        records_multi.sort(key=lambda x: x[1], reverse=True)
        best_eps, best_sil, best_labels = records_multi[0]
        best_silhouette = best_sil
        best_intra_val  = None
    else:
        # No valid clustering found at any eps
        # Fallback: use all points as static (single cluster)
        best_labels = np.zeros(n, dtype=int)
        best_eps = eps_initial
        best_silhouette = None
        best_intra_val = intra_cluster_distance(distances)
        had_multi = False

    # ── Identify static cluster ───────────────────────────────────────
    # The static cluster is the one with the MOST points
    # (background objects are persistent and have many returns)
    # In multi-cluster case, paper uses silhouette to pick best eps,
    # then the largest cluster = static background.
    unique_clusters = [l for l in set(best_labels) if l != -1]
    n_final_clusters = len(unique_clusters)

    if n_final_clusters == 0:
        # All noise — treat all as static (edge case)
        static_cluster_id = -99
        static_mask = np.ones(n, dtype=bool)
    else:
        # Static = largest cluster (most consistent distance → background)
        cluster_sizes = {c: np.sum(best_labels == c) for c in unique_clusters}
        static_cluster_id = max(cluster_sizes, key=cluster_sizes.get)
        static_mask = best_labels == static_cluster_id

    static_distances = distances[static_mask]
    outlier_mask     = ~static_mask   # True = NOT static

    # Representative static distance
    if representative == 'median':
        rep = float(np.median(static_distances)) if len(static_distances) else 0.0
    else:
        rep = float(np.mean(static_distances)) if len(static_distances) else 0.0

    return ElementClusterResult(
        channel=channel,
        azimuth_bin=azimuth_bin,
        distances=distances,
        labels=best_labels,
        best_eps=best_eps,
        best_min_pts=min_pts,
        n_clusters=n_final_clusters,
        had_multi_cluster=had_multi,
        static_cluster_id=static_cluster_id,
        static_distances=static_distances,
        static_representative=rep,
        is_outlier=outlier_mask,
        silhouette=best_silhouette,
        intra_dist=best_intra_val,
    )


# ------------------------------------------------------------------ #
#  Full matrix clustering                                              #
# ------------------------------------------------------------------ #

def cluster_all_elements(agg: np.ndarray,
                          eps_initial: float = 0.08,
                          eps_max: float = 0.40,
                          eps_step: float = 0.01,
                          min_pts_fraction: float = 0.01,
                          min_pts_floor: int = 100,
                          representative: str = 'median') -> tuple:
    """
    Apply DBSCAN clustering to every (channel, azimuth_bin) element
    of the aggregated distance matrix.

    This is the main loop described in Figure 8 (Flow Chart to
    Identify Static Scene) of the paper.

    Parameters
    ----------
    agg : np.ndarray, shape [C, M, N]
          Aggregated distance matrix from frame_extractor.

    Returns
    -------
    static_matrix : np.ndarray [C, M]
        Representative static distance for each element.
        0.0 = no static return (outlier / insufficient data).

    cluster_info : np.ndarray [C, M], dtype=object
        ElementClusterResult for each element (or None).

    stats : dict
        Summary statistics.
    """
    C, M, N = agg.shape
    total_elements = C * M

    logger.info(f"Clustering {total_elements:,} elements "
                f"[{C} channels × {M} azimuth bins]")
    logger.info(f"DBSCAN params: eps∈[{eps_initial},{eps_max}] "
                f"step={eps_step}, MinPts=max({min_pts_floor}, "
                f"{min_pts_fraction*100:.0f}% of N)")

    static_matrix = np.zeros((C, M), dtype=np.float32)
    cluster_info  = np.empty((C, M), dtype=object)

    n_static   = 0
    n_outlier  = 0
    n_nodata   = 0
    n_multi    = 0

    log_interval = max(1, total_elements // 20)

    for c in range(C):
        for m in range(M):
            elem_idx = c * M + m

            if elem_idx % log_interval == 0:
                pct = 100.0 * elem_idx / total_elements
                logger.info(f"  Progress: {elem_idx:,}/{total_elements:,} "
                            f"({pct:.1f}%) | "
                            f"static={n_static}, nodata={n_nodata}")

            # Get distances for this element
            dists = agg[c, m, :]
            dists = dists[dists > 0.0]   # remove "no return" frames

            if len(dists) < 2:
                n_nodata += 1
                cluster_info[c, m] = None
                continue

            result = cluster_element(
                distances=dists,
                channel=c,
                azimuth_bin=m,
                eps_initial=eps_initial,
                eps_max=eps_max,
                eps_step=eps_step,
                min_pts_fraction=min_pts_fraction,
                min_pts_floor=min_pts_floor,
                representative=representative,
            )

            cluster_info[c, m] = result

            if result is not None:
                static_matrix[c, m] = result.static_representative
                n_static += 1
                if result.had_multi_cluster:
                    n_multi += 1
            else:
                n_outlier += 1

    stats = {
        'total_elements':  total_elements,
        'n_static':        n_static,
        'n_outlier':       n_outlier,
        'n_nodata':        n_nodata,
        'n_multi_cluster': n_multi,
        'coverage_pct':    100.0 * n_static / total_elements,
    }

    logger.info(f"Clustering complete:")
    logger.info(f"  Static elements:       {n_static:,} ({stats['coverage_pct']:.1f}%)")
    logger.info(f"  Multi-cluster (trees): {n_multi:,}")
    logger.info(f"  No data:               {n_nodata:,}")

    return static_matrix, cluster_info, stats
