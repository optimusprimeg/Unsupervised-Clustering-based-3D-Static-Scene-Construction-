"""
static_scene.py
---------------
Implements Section 2.3 final step and Figure 17/20 of the paper.

Converts the 2D static distance matrix [C, M] back to a
3D point cloud (X, Y, Z) using the sensor geometry.

Paper Figure 2b coordinate equations:
    X = R · cos(ω) · sin(α)
    Y = R · cos(ω) · cos(α)
    Z = R · sin(ω)

Where:
    R = distance (from static_matrix)
    α = azimuth angle  (from azimuth bin index × alpha_res)
    ω = vertical angle (from laser channel index, fixed per sensor)

For VLP-16: vertical angles are fixed at
    [-15, -13, -11, -9, -7, -5, -3, -1, +1, +3, +5, +7, +9, +11, +13, +15]
    (every 2°, ordered channel 0=bottom to 15=top)

For other sensors: linearly interpolate across VFOV.
"""

import numpy as np
import logging
from src.sensor_config import SensorConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Vertical angle lookup per sensor                                    #
# ------------------------------------------------------------------ #

# VLP-16 exact vertical angles (degrees), channel 0..15
VLP16_VANGLES = np.array([
    -15, -13, -11, -9, -7, -5, -3, -1,
     +1,  +3,  +5, +7, +9,+11,+13,+15
], dtype=np.float32)

# VLP-32C vertical angles (degrees), channel 0..31
VLP32C_VANGLES = np.array([
    -25.0, -1.0,  -1.667, -15.639, -11.31,  0.0,  -0.667,  -8.843,
     -7.254, 0.333, -0.333, -6.148,  -5.333, 1.333,  0.667,  -4.0,
     -4.667, 1.667,  1.0,  -3.667,  -3.333,  3.333,  2.333, -2.667,
     -3.0,   7.0,   4.667, -2.333,  -2.0,   15.0,  10.333, -1.333
], dtype=np.float32)


def get_vertical_angles(cfg: SensorConfig) -> np.ndarray:
    """
    Return the vertical angle (omega, degrees) for each laser channel.

    Uses known sensor profiles where available; otherwise linearly
    interpolates across the full VFOV.
    """
    C = cfg.num_channels

    if C == 16:
        return VLP16_VANGLES.copy()

    if C == 32:
        return VLP32C_VANGLES.copy()

    # Generic: linear spacing from -(vfov/2) to +(vfov/2)
    half = cfg.vfov_deg / 2.0
    return np.linspace(-half, +half, C, dtype=np.float32)


# ------------------------------------------------------------------ #
#  3D reconstruction                                                   #
# ------------------------------------------------------------------ #

def reconstruct_3d(static_matrix: np.ndarray,
                   cfg: SensorConfig,
                   range_min: float = 0.1,
                   range_max: float = 100.0) -> dict:
    """
    Reconstruct the 3D static scene from the 2D static distance matrix.

    Parameters
    ----------
    static_matrix : np.ndarray [C, M]
        Static representative distance for each (channel, azimuth_bin).
        Zero = no static return.
    cfg           : SensorConfig
    range_min     : discard points closer than this (metres)
    range_max     : discard points further than this (metres)

    Returns
    -------
    dict with keys: x, y, z, distance, channel, azimuth_deg
    """
    C, M = static_matrix.shape
    assert C == cfg.num_channels, "Matrix channel count mismatch"
    assert M == cfg.num_azimuth_bins, "Matrix azimuth bin count mismatch"

    # Vertical angles per channel (degrees → radians)
    omega_deg = get_vertical_angles(cfg)          # shape [C]
    omega_rad = np.radians(omega_deg)             # shape [C]

    # Azimuth angles per bin (degrees → radians)
    alpha_deg = np.arange(M) * cfg.azimuth_resolution   # shape [M]
    alpha_rad = np.radians(alpha_deg)                    # shape [M]

    # Broadcast shapes: omega [C,1], alpha [1,M], R [C,M]
    omega = omega_rad[:, np.newaxis]   # [C, 1]
    alpha = alpha_rad[np.newaxis, :]   # [1, M]
    R     = static_matrix              # [C, M]

    # Paper Figure 2b equations
    X = R * np.cos(omega) * np.sin(alpha)
    Y = R * np.cos(omega) * np.cos(alpha)
    Z = R * np.sin(omega)

    # Valid mask: static distance is non-zero and within sensor range
    valid = (R > range_min) & (R < range_max)

    # Channel and azimuth index grids
    ch_grid  = np.tile(np.arange(C)[:, np.newaxis], (1, M))   # [C, M]
    az_grid  = np.tile(np.arange(M)[np.newaxis, :], (C, 1))   # [C, M]

    points = dict(
        x           = X[valid].astype(np.float32),
        y           = Y[valid].astype(np.float32),
        z           = Z[valid].astype(np.float32),
        distance    = R[valid].astype(np.float32),
        channel     = ch_grid[valid].astype(np.int32),
        azimuth_deg = (az_grid[valid] * cfg.azimuth_resolution).astype(np.float32),
    )

    logger.info(f"Reconstructed static scene: {len(points['x']):,} points "
                f"(from {C*M:,} elements, {100*len(points['x'])/(C*M):.1f}% coverage)")
    return points


# ------------------------------------------------------------------ #
#  Per-frame moving object extraction                                  #
# ------------------------------------------------------------------ #

def extract_moving_objects(frame: dict,
                            static_matrix: np.ndarray,
                            cfg: SensorConfig,
                            tolerance: float = 0.5) -> dict:
    """
    Given a new LiDAR frame, subtract the static scene to find
    potential moving objects.

    A point is considered a MOVING OBJECT CANDIDATE if its distance
    differs from the static background distance by more than `tolerance`
    metres at the same (channel, azimuth_bin) cell.

    This is the downstream use-case enabled by static scene construction
    (mentioned in paper Introduction and Conclusion).

    Parameters
    ----------
    frame         : normalised point cloud dict (single frame)
    static_matrix : [C, M] static distance matrix
    cfg           : SensorConfig
    tolerance     : distance difference threshold (metres)

    Returns
    -------
    dict of moving point arrays (x, y, z, distance, channel, azimuth_deg)
    """
    from src.frame_extractor import (azimuth_to_bin, assign_ring_ids)

    az_deg   = np.degrees(np.arctan2(frame['y'], frame['x'])) % 360.0
    az_bins  = azimuth_to_bin(az_deg, cfg.azimuth_resolution, cfg.num_azimuth_bins)
    ring_ids = assign_ring_ids(frame, cfg)
    r        = frame['distance']

    valid = (r > 0.1) & (r < 100.0) & (ring_ids >= 0) & (ring_ids < cfg.num_channels)

    r_v  = r[valid]
    az_v = az_bins[valid]
    ch_v = ring_ids[valid]

    # Lookup static distance for each point's cell
    static_r = static_matrix[ch_v, az_v]

    # Moving if: static_r exists (>0) AND |r - static_r| > tolerance
    #         OR: static_r exists but point is closer (object in foreground)
    moving_mask = (static_r > 0.0) & (np.abs(r_v - static_r) > tolerance)

    # Reconstruct 3D for moving points
    omega_deg = get_vertical_angles(cfg)
    omega_rad = np.radians(omega_deg)
    alpha_rad = np.radians(az_v[moving_mask] * cfg.azimuth_resolution)
    omega_r   = omega_rad[ch_v[moving_mask]]
    R_m       = r_v[moving_mask]

    return dict(
        x           = (R_m * np.cos(omega_r) * np.sin(alpha_rad)).astype(np.float32),
        y           = (R_m * np.cos(omega_r) * np.cos(alpha_rad)).astype(np.float32),
        z           = (R_m * np.sin(omega_r)).astype(np.float32),
        distance    = R_m.astype(np.float32),
        channel     = ch_v[moving_mask].astype(np.int32),
        azimuth_deg = (az_v[moving_mask] * cfg.azimuth_resolution).astype(np.float32),
    )
