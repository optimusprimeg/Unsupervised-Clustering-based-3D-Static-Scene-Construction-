"""
sensor_config.py
----------------
Auto-detects LiDAR sensor configuration from loaded PCD frames.

Detects:
  - Number of laser channels (C)
  - Azimuth resolution (alpha_res) in degrees
  - Vertical angles per channel (omega)
  - Total elements per frame = C * (360 / alpha_res)

Paper reference: Section 2.2 (Data Extraction), Equation 2
    Number of Elements = C × (360° / alpha_res)
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Known sensor profiles for sanity-check / fallback
_KNOWN_SENSORS = {
    16:  {'name': 'VLP-16',  'alpha_res': 0.2,  'vfov': (-15.0, 15.0),  'vres': 2.0},
    32:  {'name': 'VLP-32',  'alpha_res': 0.2,  'vfov': (-25.0, 15.0),  'vres': None},
    64:  {'name': 'HDL-64E', 'alpha_res': 0.08, 'vfov': (-24.8, 2.0),   'vres': None},
    128: {'name': 'VLS-128', 'alpha_res': 0.1,  'vfov': (-25.0, 15.0),  'vres': None},
}


class SensorConfig:
    """Holds all auto-detected or user-supplied sensor parameters."""

    def __init__(self,
                 num_channels: int,
                 azimuth_resolution: float,
                 vfov_deg: float = 30.0,
                 sensor_name: str = "unknown"):
        self.num_channels       = num_channels           # C
        self.azimuth_resolution = azimuth_resolution     # alpha_res (degrees)
        self.vfov_deg           = vfov_deg
        self.sensor_name        = sensor_name

        # Derived (paper Equation 2)
        self.num_azimuth_bins   = int(round(360.0 / azimuth_resolution))
        self.num_elements       = num_channels * self.num_azimuth_bins

    def __repr__(self):
        return (f"SensorConfig(sensor={self.sensor_name}, "
                f"channels={self.num_channels}, "
                f"alpha_res={self.azimuth_resolution:.3f}°, "
                f"azimuth_bins={self.num_azimuth_bins}, "
                f"total_elements={self.num_elements})")


def detect_num_channels(frames: list,
                        forced: Optional[int] = None) -> int:
    """
    Detect number of laser channels from the 'ring' field.

    Strategy:
      1. If forced is not None, return it directly.
      2. If ring field is populated (not all -1), use max(ring)+1.
      3. Otherwise, estimate from vertical angle (z/R) distribution.
    """
    if forced is not None:
        logger.info(f"Sensor channels forced to {forced}")
        return forced

    # Try ring field
    ring_vals = []
    for f in frames[:min(50, len(frames))]:
        rings = f['ring']
        valid = rings[rings >= 0]
        if len(valid):
            ring_vals.append(int(valid.max()))

    if ring_vals:
        C = max(ring_vals) + 1
        logger.info(f"Auto-detected {C} channels from 'ring' field.")
        return C

    # Fallback: cluster vertical angles
    logger.info("'ring' field absent — estimating channels from vertical angles.")
    all_z, all_r = [], []
    for f in frames[:min(20, len(frames))]:
        r = f['distance']
        valid = r > 0.1
        all_z.append(f['z'][valid])
        all_r.append(r[valid])

    z = np.concatenate(all_z)
    r = np.concatenate(all_r)
    valid = r > 0.1
    el_deg = np.degrees(np.arcsin(np.clip(z[valid] / r[valid], -1, 1)))

    # histogram to find discrete elevation levels
    hist, edges = np.histogram(el_deg, bins=500)
    # find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hist, height=len(z) * 0.001, distance=2)
    C = len(peaks)
    if C == 0:
        C = 16  # safe fallback
        logger.warning(f"Could not detect channels; defaulting to {C}")
    else:
        logger.info(f"Auto-detected {C} channels from elevation histogram.")

    # round to nearest known sensor
    known = sorted(_KNOWN_SENSORS.keys())
    C = min(known, key=lambda k: abs(k - C))
    logger.info(f"Rounded to nearest known sensor: {C} channels.")
    return C


def detect_azimuth_resolution(frames: list,
                               forced: Optional[float] = None) -> float:
    """
    Detect azimuth resolution by analysing the distribution of azimuth
    angles within a frame.

    Strategy:
      1. Compute per-point azimuth = atan2(y, x) in degrees [0, 360).
      2. Sort and compute differences.
      3. Most common non-zero difference ≈ alpha_res.
    """
    if forced is not None:
        logger.info(f"Azimuth resolution forced to {forced:.3f}°")
        return forced

    # Use a few frames
    diffs = []
    for f in frames[:min(10, len(frames))]:
        az = np.degrees(np.arctan2(f['y'], f['x'])) % 360.0
        az_sorted = np.sort(np.unique(az.round(4)))
        d = np.diff(az_sorted)
        d = d[(d > 0.01) & (d < 2.0)]   # filter wrap-arounds & noise
        diffs.append(d)

    if not diffs:
        alpha_res = 0.2
        logger.warning(f"Could not detect azimuth resolution; defaulting to {alpha_res}°")
        return alpha_res

    all_diffs = np.concatenate(diffs)
    # bin at 0.01° precision
    hist, edges = np.histogram(all_diffs, bins=np.arange(0.0, 2.0, 0.005))
    alpha_res = edges[np.argmax(hist)] + 0.0025   # bin centre

    # snap to nearest known value
    known_res = [0.08, 0.1, 0.16, 0.18, 0.2, 0.33, 0.4]
    alpha_res = min(known_res, key=lambda r: abs(r - alpha_res))
    logger.info(f"Auto-detected azimuth resolution: {alpha_res:.3f}°")
    return alpha_res


def auto_detect(frames: list,
                forced_channels: Optional[int] = None,
                forced_alpha_res: Optional[float] = None,
                vfov_deg: float = 30.0) -> SensorConfig:
    """
    Full auto-detection pipeline.

    Parameters
    ----------
    frames          : list of normalised point-cloud dicts
    forced_channels : override channel count (from config)
    forced_alpha_res: override azimuth resolution (from config)
    vfov_deg        : total vertical FOV

    Returns
    -------
    SensorConfig
    """
    C         = detect_num_channels(frames, forced_channels)
    alpha_res = detect_azimuth_resolution(frames, forced_alpha_res)

    known = _KNOWN_SENSORS.get(C, {})
    name  = known.get('name', f"Unknown-{C}ch")

    cfg = SensorConfig(C, alpha_res, vfov_deg, name)
    logger.info(f"Sensor configuration: {cfg}")
    return cfg
