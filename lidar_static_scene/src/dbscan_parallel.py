"""
dbscan_parallel.py
------------------
Parallel wrapper around cluster_all_elements().

Uses Python multiprocessing to distribute the (C × M) element-wise
DBSCAN work across CPU cores.  Exact same algorithm as dbscan_clustering.py;
only the execution strategy changes.

Usage:
    from src.dbscan_parallel import cluster_all_elements_parallel
    static_matrix, cluster_info, stats = cluster_all_elements_parallel(agg, cfg, n_jobs=-1)

n_jobs: number of worker processes (-1 = all cores, 1 = serial fallback)
"""

import numpy as np
import multiprocessing as mp
from functools import partial
import logging
import time

from src.dbscan_clustering import cluster_element, ElementClusterResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Worker function (must be top-level for pickling)                    #
# ------------------------------------------------------------------ #

def _worker(args):
    """
    Process one (channel, azimuth_bin) element.
    args = (c, m, distances_1d, kwargs_dict)
    Returns (c, m, ElementClusterResult or None)
    """
    c, m, dists, kwargs = args
    if len(dists) < 2:
        return (c, m, None)
    result = cluster_element(dists, channel=c, azimuth_bin=m, **kwargs)
    return (c, m, result)


# ------------------------------------------------------------------ #
#  Parallel driver                                                     #
# ------------------------------------------------------------------ #

def cluster_all_elements_parallel(agg: np.ndarray,
                                   eps_initial:      float = 0.08,
                                   eps_max:          float = 0.40,
                                   eps_step:         float = 0.01,
                                   min_pts_fraction: float = 0.01,
                                   min_pts_floor:    int   = 100,
                                   representative:   str   = 'median',
                                   n_jobs:           int   = -1,
                                   chunk_size:       int   = 64) -> tuple:
    """
    Parallel version of cluster_all_elements().

    Parameters
    ----------
    agg         : np.ndarray [C, M, N]
    n_jobs      : worker processes (-1 = cpu_count, 1 = serial)
    chunk_size  : elements per worker task (tune for memory vs overhead)

    Returns
    -------
    static_matrix : np.ndarray [C, M]
    cluster_info  : np.ndarray [C, M] dtype=object (ElementClusterResult|None)
    stats         : dict
    """
    C, M, N = agg.shape
    total   = C * M

    n_cpus = mp.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    logger.info(f"Parallel DBSCAN: {total:,} elements across {n_cpus} workers")
    logger.info(f"DBSCAN params: eps∈[{eps_initial},{eps_max}] step={eps_step}, "
                f"MinPts=max({min_pts_floor}, {min_pts_fraction*100:.0f}% of N)")

    kwargs = dict(
        eps_initial      = eps_initial,
        eps_max          = eps_max,
        eps_step         = eps_step,
        min_pts_fraction = min_pts_fraction,
        min_pts_floor    = min_pts_floor,
        representative   = representative,
    )

    # Build task list: (c, m, dists_1d, kwargs)
    tasks = []
    for c in range(C):
        for m in range(M):
            dists = agg[c, m, :]
            dists = dists[dists > 0.0]
            tasks.append((c, m, dists, kwargs))

    static_matrix = np.zeros((C, M), dtype=np.float32)
    cluster_info  = np.empty((C, M), dtype=object)

    n_static  = 0
    n_outlier = 0
    n_nodata  = 0
    n_multi   = 0

    t0 = time.time()
    log_interval = max(1, total // 20)

    if n_cpus == 1:
        # Serial fallback
        for i, task in enumerate(tasks):
            c, m, result_tuple = _worker(task)
            _, _, result = c, m, result_tuple  # unpack
            c2, m2, result = _worker(task)
            _record(result, c2, m2, static_matrix, cluster_info,
                    n_static, n_nodata, n_multi, n_outlier)
            if i % log_interval == 0:
                pct = 100.0 * i / total
                elapsed = time.time() - t0
                eta = elapsed / max(i, 1) * (total - i)
                logger.info(f"  {i:,}/{total:,} ({pct:.1f}%) "
                            f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
    else:
        completed = 0
        with mp.Pool(processes=n_cpus) as pool:
            for c, m, result in pool.imap_unordered(
                    _worker, tasks, chunksize=chunk_size):
                if result is None:
                    n_nodata += 1
                else:
                    static_matrix[c, m] = result.static_representative
                    n_static += 1
                    if result.had_multi_cluster:
                        n_multi += 1
                cluster_info[c, m] = result
                completed += 1

                if completed % log_interval == 0:
                    pct = 100.0 * completed / total
                    elapsed = time.time() - t0
                    eta = elapsed / completed * (total - completed)
                    logger.info(f"  {completed:,}/{total:,} ({pct:.1f}%) "
                                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    stats = {
        'total_elements':  total,
        'n_static':        n_static,
        'n_outlier':       n_outlier,
        'n_nodata':        n_nodata,
        'n_multi_cluster': n_multi,
        'coverage_pct':    100.0 * n_static / total,
    }

    elapsed = time.time() - t0
    logger.info(f"Parallel clustering done in {elapsed:.1f}s")
    logger.info(f"  Static: {n_static:,} ({stats['coverage_pct']:.1f}%)  "
                f"Multi-cluster: {n_multi:,}  No-data: {n_nodata:,}")

    return static_matrix, cluster_info, stats


def _record(result, c, m, static_matrix, cluster_info,
            n_static, n_nodata, n_multi, n_outlier):
    """Helper to record a single result into output arrays."""
    cluster_info[c, m] = result
    if result is None:
        n_nodata += 1
    else:
        static_matrix[c, m] = result.static_representative
        n_static += 1
        if result.had_multi_cluster:
            n_multi += 1
