"""
pcd_reader.py
-------------
Standalone PCD file reader (ASCII, binary, binary_compressed).
No open3d required. Returns numpy structured arrays.

Handles A9-dataset PCD files which may contain fields:
  x y z intensity ring (channel) timestamp  -- or subset thereof
"""

import struct
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

_DTYPE_MAP = {
    ('F', 4): np.float32,
    ('F', 8): np.float64,
    ('I', 1): np.int8,
    ('I', 2): np.int16,
    ('I', 4): np.int32,
    ('I', 8): np.int64,
    ('U', 1): np.uint8,
    ('U', 2): np.uint16,
    ('U', 4): np.uint32,
    ('U', 8): np.uint64,
}


def _parse_header(fp):
    """Read PCD header lines and return a dict of metadata."""
    header = {}
    while True:
        line = fp.readline()
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='replace')
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, *vals = line.split()
        key = key.upper()
        header[key] = vals
        if key == 'DATA':
            break
    return header


def _build_dtype(fields, sizes, types, counts):
    """Build numpy dtype from PCD header field descriptors."""
    dt_fields = []
    for f, s, t, c in zip(fields, sizes, types, counts):
        base = _DTYPE_MAP.get((t.upper(), int(s)))
        if base is None:
            base = np.float32
        c = int(c)
        if c == 1:
            dt_fields.append((f, base))
        else:
            dt_fields.append((f, base, (c,)))
    return np.dtype(dt_fields)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def read_pcd(path: str) -> np.ndarray:
    """
    Read a .pcd file and return a structured numpy array.

    Returns
    -------
    np.ndarray with named fields matching the PCD FIELDS header.
    Guaranteed fields after normalisation:
        x, y, z  (float32)
    Optional fields (present if in file):
        intensity / i  (float32)
        ring / channel (uint16 or uint8)
        timestamp      (float64 or uint32)
    """
    with open(path, 'rb') as fp:
        header = _parse_header(fp)
        data_start = fp.tell()
        raw = fp.read()

    fields  = header.get('FIELDS', ['x', 'y', 'z'])
    sizes   = header.get('SIZE',   ['4', '4', '4'])
    types   = header.get('TYPE',   ['F', 'F', 'F'])
    counts  = header.get('COUNT',  ['1'] * len(fields))
    width   = int(header.get('WIDTH',  ['0'])[0])
    height  = int(header.get('HEIGHT', ['1'])[0])
    points  = int(header.get('POINTS', [str(width * height)])[0])
    enc     = header.get('DATA', ['ascii'])[0].lower()

    dtype = _build_dtype(fields, sizes, types, counts)

    if enc == 'ascii':
        # ASCII: whitespace-delimited rows
        txt = raw.decode('utf-8', errors='replace')
        rows = [row.split() for row in txt.strip().splitlines() if row.strip()]
        arr = np.array(rows, dtype=np.float64)
        cloud = np.zeros(len(arr), dtype=dtype)
        for i, f in enumerate(fields):
            try:
                cloud[f] = arr[:, i].astype(dtype[f].base)
            except Exception:
                pass

    elif enc == 'binary':
        cloud = np.frombuffer(raw[:points * dtype.itemsize], dtype=dtype)

    elif enc == 'binary_compressed':
        # LZF compression: first 8 bytes = compressed_size, uncompressed_size
        comp_size   = struct.unpack_from('<I', raw, 0)[0]
        uncomp_size = struct.unpack_from('<I', raw, 4)[0]
        compressed  = raw[8: 8 + comp_size]
        try:
            import lzf as _lzf
            uncompressed = _lzf.decompress(compressed, uncomp_size)
        except ImportError:
            raise RuntimeError(
                "binary_compressed PCD requires the 'lzf' package. "
                "Install it with: pip install lzf"
            )
        # PCD binary_compressed stores columns (not rows): de-columnarise
        cloud = np.zeros(points, dtype=dtype)
        offset = 0
        for f, s, t, c in zip(fields, sizes, types, counts):
            c = int(c)
            nbytes = int(s) * c * points
            col_data = np.frombuffer(uncompressed[offset:offset + nbytes],
                                     dtype=_DTYPE_MAP.get((t.upper(), int(s)), np.float32))
            if c == 1:
                cloud[f] = col_data
            else:
                cloud[f] = col_data.reshape(points, c)
            offset += nbytes
    else:
        raise ValueError(f"Unknown PCD encoding: {enc}")

    return cloud


def normalise_cloud(cloud: np.ndarray) -> dict:
    """
    Convert a structured PCD array into a plain dict of float32 arrays.
    Normalises common field-name variants.

    Returns dict with keys: x, y, z, distance, intensity, ring
    'ring'     = laser channel id  (0-indexed)
    'distance' = R = sqrt(x^2 + y^2 + z^2)
    """
    names = cloud.dtype.names

    def _get(candidates, default=None):
        for c in candidates:
            if c in names:
                return cloud[c].astype(np.float32)
        return default

    x = _get(['x'], np.zeros(len(cloud), np.float32))
    y = _get(['y'], np.zeros(len(cloud), np.float32))
    z = _get(['z'], np.zeros(len(cloud), np.float32))

    intensity = _get(['intensity', 'i', 'reflectivity'],
                     np.zeros(len(cloud), np.float32))

    ring = _get(['ring', 'channel', 'laser_id', 'layer'],
                np.full(len(cloud), -1, np.float32))

    distance = np.sqrt(x**2 + y**2 + z**2).astype(np.float32)

    return dict(x=x, y=y, z=z,
                intensity=intensity,
                ring=ring.astype(np.int32),
                distance=distance)


def load_pcd_frames(pcd_dir: str,
                    max_frames: int = None,
                    step: int = 1) -> list:
    """
    Load all .pcd files from a directory as a list of point-cloud dicts.

    Parameters
    ----------
    pcd_dir   : directory containing .pcd files
    max_frames: stop after this many frames (None = all)
    step      : take every `step`-th file

    Returns
    -------
    List of dicts, each from normalise_cloud()
    """
    files = sorted([
        os.path.join(pcd_dir, f)
        for f in os.listdir(pcd_dir)
        if f.lower().endswith('.pcd')
    ])

    if not files:
        raise FileNotFoundError(f"No .pcd files found in: {pcd_dir}")

    files = files[::step]
    if max_frames is not None:
        files = files[:max_frames]

    logger.info(f"Loading {len(files)} PCD frames from {pcd_dir}")

    frames = []
    for i, path in enumerate(files):
        try:
            cloud = read_pcd(path)
            normed = normalise_cloud(cloud)
            frames.append(normed)
            if (i + 1) % 500 == 0:
                logger.info(f"  Loaded {i+1}/{len(files)} frames")
        except Exception as e:
            logger.warning(f"  Skipping {path}: {e}")

    logger.info(f"Successfully loaded {len(frames)} frames.")
    return frames
