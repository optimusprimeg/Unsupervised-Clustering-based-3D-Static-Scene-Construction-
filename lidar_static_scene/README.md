# Unsupervised Clustering-based 3D Static Scene Construction
### Paper Replication — Rajput et al., ISPRS Archives 2024

> **"Unsupervised Clustering-based 3D Static Scene Construction Using LiDAR Channel and Azimuth Angle"**  
> Rohit Rajput, Salil Goel, Aditya Medury — IIT Kanpur  
> ISPRS TC IV Mid-term Symposium, Perth, Australia, October 2024  
> DOI: https://doi.org/10.5194/isprs-archives-XLVIII-4-2024-381-2024

---

## Overview

This repository is a **complete, exact replication** of the paper's algorithm.
Every parameter, formula, and decision rule is taken directly from the paper text.

The algorithm constructs a **3D static background scene** from a roadside LiDAR
(e.g. Velodyne VLP-16) using only unsupervised DBSCAN clustering — no labels,
no training data, no manual parameter tuning.

---

## Algorithm Summary (Paper §2)

```
Input: N LiDAR frames (.pcd files)

For each frame:
  └─ Build 2D distance matrix [C rows × M cols]
       C = num laser channels
       M = 360° / azimuth_resolution          ← Paper Eq. 2

Aggregate distances across all N frames
  └─ Result: [C, M, N] tensor

For each element (c, m):                       ← Paper §2.3.1
  └─ Distances = agg[c, m, :]  (non-zero only)
  └─ MinPts = max(100, 1% × len(distances))
  └─ Sweep eps from 0.08 → 0.40 (step 0.01):
       If multiple clusters → track MAX silhouette score
       If single cluster   → track MIN intra-cluster distance
  └─ Apply DBSCAN with best eps
  └─ Static cluster = largest cluster
  └─ Representative distance = median

Reconstruct 3D:                                ← Paper Fig. 2b
  X = R · cos(ω) · sin(α)
  Y = R · cos(ω) · cos(α)
  Z = R · sin(ω)

Output: static_scene.pcd
```

---

## Project Structure

```
lidar_static_scene/
├── main.py                    # Entry point — full pipeline
├── config.yaml                # All parameters documented
├── src/
│   ├── pcd_reader.py          # ASCII/binary/binary_compressed PCD reader
│   ├── sensor_config.py       # Auto-detect channels & azimuth resolution
│   ├── frame_extractor.py     # Per-frame 2D distance matrix builder
│   ├── dbscan_clustering.py   # Paper §2.3.1 — exact DBSCAN per element
│   ├── dbscan_parallel.py     # Multiprocessing wrapper (same algorithm)
│   ├── static_scene.py        # 3D reconstruction + moving object extraction
│   └── visualizer.py          # Figures matching paper + PCD writer
├── data/                      # Put your .pcd files here
└── output/                    # All results written here
```

---

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib tqdm
# open3d is NOT required — PCD reading is built-in
```

Optional GPU acceleration (same algorithm, faster DBSCAN runtime):

```bash
# Requires NVIDIA GPU + CUDA-compatible RAPIDS install
pip install cupy-cuda12x cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

---

## Usage

### 1. Quick demo (no dataset needed)

```bash
python main.py --demo
```

Generates synthetic VLP-16 data with a static background and moving objects,
runs the full pipeline, and saves all outputs to `output/`.

### 2. Real A9 dataset (or any roadside LiDAR .pcd files)

```bash
# Auto-detect everything
python main.py --pcd_dir /path/to/a9/pcd_files/

# Prefer GPU if available, else fallback to CPU path
python main.py --pcd_dir /path/to/a9/pcd_files/ --compute_backend auto

# Force CPU multiprocessing
python main.py --pcd_dir /path/to/a9/pcd_files/ --compute_backend cpu --n_jobs -1

# Force GPU backend (fails if RAPIDS/CUDA is unavailable)
python main.py --pcd_dir /path/to/a9/pcd_files/ --compute_backend gpu

# Limit frames for a quick test
python main.py --pcd_dir /path/to/a9/pcd_files/ --max_frames 500

# If sensor is known, skip auto-detection
python main.py --pcd_dir /path/to/a9/pcd_files/ --channels 32 --alpha_res 0.2

# Load every 3rd frame (speeds up loading from large datasets)
python main.py --pcd_dir /path/to/a9/pcd_files/ --frame_step 3
```

### 3. Paper-exact parameters (already the default)

```bash
python main.py \
  --pcd_dir data/ \
  --eps_initial 0.08 \
  --eps_max 0.40 \
  --eps_step 0.01
```

---

## A9 Dataset Notes

The **A9-Dataset** (TU Munich roadside perception dataset) is directly compatible.

| Paper Setup | A9 Dataset | Notes |
|---|---|---|
| VLP-16 (16 ch) | Varies (32/64 ch) | Auto-detected |
| 0.2° azimuth res | Sensor-dependent | Auto-detected |
| `.bag` format | `.pcd` format | Built-in reader handles this |
| 4-legged intersection | Multiple locations | Direct match |
| Static sensor mount | Infrastructure-mounted | Direct match |

**A9 download:** https://a9-dataset.com  
**Relevant split:** Use `infrastructure_side` point clouds only  
(vehicle-side LiDAR will not have a static background)

---

## Outputs

All outputs written to `output/`:

| File | Description |
|---|---|
| `static_scene.pcd` | 3D static background point cloud |
| `outlier_points.pcd` | Moving object candidates (reference frame) |
| `static_distance_matrix.npy` | [C × M] numpy array of static distances |
| `aggregated_matrix.npy` | [C × M × N] raw aggregated distances |
| `fig20_static_scene_combined.png` | 3D scatter (matches paper Fig. 20) |
| `fig17_static_with_outliers.png` | Static=white, Outlier=red (paper Fig. 17) |
| `static_distance_matrix_heatmap.png` | [C × M] distance heatmap |
| `fig15_*_dbscan.png` | DBSCAN output per element (paper Fig. 15/16) |
| `fig12_*_dist_scatter.png` | Distance vs frame scatter (paper Fig. 12/14) |
| `results_summary.txt` | Statistics report |

---

## Performance

The bottleneck is the per-element DBSCAN loop: **C × M elements** total.

| Sensor | Elements | Serial (est.) | 8-core parallel |
|---|---|---|---|
| VLP-16 (16ch, 0.2°) | 28,800 | ~2.5 hrs | ~20 min |
| VLP-32 (32ch, 0.2°) | 57,600 | ~5 hrs | ~40 min |
| HDL-64 (64ch, 0.08°) | 288,000 | ~25 hrs | ~3.5 hrs |

**Speedup options:**

```bash
# Use all CPU cores (default)
python main.py --pcd_dir data/ --compute_backend cpu --n_jobs -1

# Reduce frame count (more frames = better static estimate, but slower)
python main.py --pcd_dir data/ --max_frames 1000

# Wider eps step (less eps values to sweep, paper uses 0.01)
python main.py --pcd_dir data/ --eps_step 0.02
```

---

## Paper Implementation Fidelity

Every detail from the paper is implemented exactly:

| Paper statement | Implementation |
|---|---|
| "MinPts = max(100, 1% × N)" | `dbscan_clustering.py:cluster_element()` line ~100 |
| "eps from 0.08 to 0.4, step 0.01" | Same function, `eps_values = np.arange(...)` |
| "max silhouette → multi-cluster" | `records_multi.sort(key=lambda x: x[1], reverse=True)` |
| "min intra-dist → single cluster" | `records_single.sort(key=lambda x: x[1])` |
| "Arc Length S = D·θ (3.49 mm/m)" | Documented in `config.yaml`, used in eps_initial |
| "N elements = C × (360°/α_res)" | `sensor_config.py:SensorConfig.num_elements` |
| "X=R·cos(ω)·sin(α)" etc. | `static_scene.py:reconstruct_3d()` |
| VLP-16 exact vertical angles | `static_scene.py:VLP16_VANGLES` |

---

## Citation

```bibtex
@article{rajput2024static,
  title   = {Unsupervised Clustering-based 3D Static Scene Construction
             Using LiDAR Channel and Azimuth Angle},
  author  = {Rajput, Rohit and Goel, Salil and Medury, Aditya},
  journal = {The International Archives of the Photogrammetry,
             Remote Sensing and Spatial Information Sciences},
  volume  = {XLVIII-4-2024},
  pages   = {381--387},
  year    = {2024},
  doi     = {10.5194/isprs-archives-XLVIII-4-2024-381-2024}
}
```
