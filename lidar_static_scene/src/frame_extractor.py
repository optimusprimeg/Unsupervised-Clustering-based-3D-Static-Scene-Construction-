"""
frame_extractor.py
------------------
Implements Section 2.2 (Data Extraction) of the paper exactly.

Key steps:
  1. For each frame, assign each point to its (laser_id, azimuth_bin) cell.
  2. Build a 2D distance matrix: shape [C, M]
       C = num_channels
       M = num_azimuth_bins = 360 / alpha_res
  3. Each cell stores the distance R for that (laser_id, azimuth_bin) pair.
     If multiple returns exist (rare for single-return sensors), take nearest.

Paper quote:
  "We extract the point distance for each azimuth angle and laser channels.
   This information is used to create a 2D matrix with N rows, representing
   the number of laser channels (C) and M columns represents azimuth values
   (α), effectively spanning 360° at an interval of 0.2°."

Paper Equation 2:
  Number of Elements = C × (360° / alpha_res)
  VLP-16 @ 0.2° => 16 × 1800 = 28,800 elements
"""

import numpy as np
import logging
from typing import Optional
from src.sensor_config import SensorConfig

logger = logging.getLogger(__name__)


def azimuth_to_bin(azimuth_deg: np.ndarray,
                   alpha_res: float,
                   num_bins: int) -> np.ndarray:
    """
    Map continuous azimuth angles (degrees, [0,360)) to integer bin indices.

    Parameters
    ----------
    azimuth_deg : array of azimuth values in [0, 360)
    alpha_res   : azimuth resolution in degrees
    num_bins    : total number of bins = 360 / alpha_res

    Returns
    -------
    Integer bin indices in [0, num_bins)
    """
    bins = (azimuth_deg / alpha_res).astype(np.int32)
    bins = np.clip(bins, 0, num_bins - 1)
    return bins


def assign_ring_ids(frame: dict,
                    cfg: SensorConfig) -> np.ndarray:
    """
    Return ring (laser channel) ids for each point.
    If 'ring' field is present and valid, use it.
    Otherwise, estimate from vertical angle (elevation).

    Paper uses the laser channel directly from sensor output.
    VLP-16 outputs ring 0..15 ordered by vertical angle.
    """
    rings = frame['ring']

    if np.any(rings >= 0):
        # Valid ring field — use directly
        return np.clip(rings, 0, cfg.num_channels - 1)

    # Estimate from elevation angle
    r = frame['distance']
    z = frame['z']
    valid = r > 1e-3
    el_deg = np.full(len(r), 0.0, dtype=np.float32)
    el_deg[valid] = np.degrees(np.arcsin(
        np.clip(z[valid] / r[valid], -1.0, 1.0)
    ))

    # VLP-16 vertical angles: -15, -13, -11, ..., +13, +15 (2° steps)
    # General: linearly map [-vfov/2, +vfov/2] to [0, C-1]
    half_vfov = cfg.vfov_deg / 2.0
    ring_float = (el_deg + half_vfov) / cfg.vfov_deg * (cfg.num_channels - 1)
    ring_ids = np.round(ring_float).astype(np.int32)
    ring_ids = np.clip(ring_ids, 0, cfg.num_channels - 1)
    return ring_ids


def build_frame_distance_matrix(frame: dict,
                                 cfg: SensorConfig) -> np.ndarray:
    """
    Build the elementwise 2D distance matrix for ONE frame.

    Shape: [C, M]  where C=num_channels, M=num_azimuth_bins

    Each cell (c, m) = distance R of the point fired at
    laser channel c and azimuth bin m.
    If no point hit that cell, value = 0.0 (no return).
    If multiple points in same cell, keep the closest (min distance).

    This is Figure 6 in the paper.
    """
    C = cfg.num_channels
    M = cfg.num_azimuth_bins

    # Compute azimuth angle per point: atan2(y, x) mapped to [0, 360)
    az_deg = np.degrees(np.arctan2(frame['y'], frame['x'])) % 360.0
    az_bins = azimuth_to_bin(az_deg, cfg.azimuth_resolution, M)

    # Get ring ids
    ring_ids = assign_ring_ids(frame, cfg)

    # Filter: valid range
    r = frame['distance']
    valid_mask = (r > 0.1) & (r < 100.0) & (ring_ids >= 0) & (ring_ids < C)

    r        = r[valid_mask]
    az_bins  = az_bins[valid_mask]
    ring_ids = ring_ids[valid_mask]

    # Build matrix — for duplicate (c,m): keep minimum distance
    dist_matrix = np.zeros((C, M), dtype=np.float32)

    # Use np.minimum.at for correct min-accumulation
    # Flatten to 1D index: idx = ring * M + az_bin
    flat_idx = ring_ids * M + az_bins
    # Sort by distance descending so minimum ends up last (overwrite)
    sort_order = np.argsort(r)[::-1]   # descending: large written first
    np.add.at(dist_matrix.ravel(), flat_idx[sort_order], 0)   # init
    # Actually do it cleanly with a loop-free scatter-min:
    dist_matrix_flat = dist_matrix.ravel()
    # Write all, then keep min: use temporary large array
    temp = np.full(C * M, np.inf, dtype=np.float32)
    np.minimum.at(temp, flat_idx, r)
    temp[temp == np.inf] = 0.0
    dist_matrix = temp.reshape(C, M)

    return dist_matrix


def build_aggregated_distance_matrix(frames: list,
                                      cfg: SensorConfig) -> np.ndarray:
    """
    Aggregate the per-frame distance matrices across ALL frames.

    Result shape: [C, M, N_frames]
      where N_frames is the number of frames with a valid return
      at each (c, m) cell.

    This is Figure 7 in the paper — the "elementwise aggregated
    distance matrix".  Each cell (c, m) holds a LIST of distances
    observed across frames; DBSCAN is then applied per element.

    To avoid a Python list-of-lists, we store as a 3D numpy array
    of shape [C, M, N] where N = len(frames).  Zero entries mean
    "no return in that frame" and are filtered out before DBSCAN.

    Returns
    -------
    agg : np.ndarray, shape [C, M, N], dtype float32
    """
    C = cfg.num_channels
    M = cfg.num_azimuth_bins
    N = len(frames)

    logger.info(f"Building aggregated distance matrix: "
                f"[{C} channels × {M} azimuth bins × {N} frames]")
    logger.info(f"Total elements to cluster: {C * M:,}")

    agg = np.zeros((C, M, N), dtype=np.float32)

    for i, frame in enumerate(frames):
        dm = build_frame_distance_matrix(frame, cfg)   # [C, M]
        agg[:, :, i] = dm
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1}/{N} frames")

    logger.info("Aggregation complete.")
    return agg


def get_element_distances(agg: np.ndarray,
                           c: int,
                           m: int) -> np.ndarray:
    """
    Retrieve all non-zero distances across frames for element (c, m).
    Returns a 1D array of valid distances.
    """
    dists = agg[c, m, :]
    return dists[dists > 0.0]
