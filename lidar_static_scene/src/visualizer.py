"""
visualizer.py
-------------
Visualization and file I/O for the static scene pipeline.

Generates:
  1. ASCII .pcd files for static scene and outlier points
  2. Matplotlib figures matching paper figures:
       - Figure 17: Static scene (single frame overlay)
       - Figure 18: Outlier points in single frame
       - Figure 20: Combined frames static scene
       - Figure 12/14: Distance-vs-frame scatter for an element
       - Figure 15/16: DBSCAN cluster output for an element
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import logging

logger = logging.getLogger(__name__)


def _axis_limits(values: np.ndarray, pad_ratio: float = 0.05):
    """Return padded min/max limits for plotting."""
    if len(values) == 0:
        return (-1.0, 1.0)
    lo = float(np.percentile(values, 1))
    hi = float(np.percentile(values, 99))
    pad = max(0.1, (hi - lo) * pad_ratio)
    return lo - pad, hi + pad


def _style_scene_axis(ax, x, y, z, elev: float = 18, azim: float = -62):
    """Apply paper-like 3D styling."""
    ax.set_facecolor('black')
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('persp')
    ax.grid(False)
    ax.tick_params(colors='white', labelsize=10)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('gray')

    xlim = _axis_limits(x)
    ylim = _axis_limits(y)
    zlim = _axis_limits(z)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    try:
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────── #
#  PCD file writer                                                     #
# ─────────────────────────────────────────────────────────────────── #

def write_pcd(path: str, points: dict, extra_fields: list = None):
    """
    Write a point cloud dict to ASCII PCD format.

    Parameters
    ----------
    path         : output file path
    points       : dict with keys x, y, z (required), plus optionals
    extra_fields : list of additional field names to include
    """
    x = points['x']
    y = points['y']
    z = points['z']
    n = len(x)

    fields = ['x', 'y', 'z']
    if extra_fields:
        fields += [f for f in extra_fields if f in points]

    sizes  = ['4'] * len(fields)
    types  = ['F'] * len(fields)
    counts = ['1'] * len(fields)

    with open(path, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data\n")
        f.write(f"VERSION 0.7\n")
        f.write(f"FIELDS {' '.join(fields)}\n")
        f.write(f"SIZE {' '.join(sizes)}\n")
        f.write(f"TYPE {' '.join(types)}\n")
        f.write(f"COUNT {' '.join(counts)}\n")
        f.write(f"WIDTH {n}\n")
        f.write(f"HEIGHT 1\n")
        f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write(f"DATA ascii\n")

        data = np.column_stack([points[fld].astype(np.float32)
                                for fld in fields])
        np.savetxt(f, data, fmt='%.6f')

    logger.info(f"Wrote {n:,} points → {path}")


def save_distance_matrix(path: str, matrix: np.ndarray):
    """Save [C, M] static distance matrix as numpy binary."""
    np.save(path, matrix)
    logger.info(f"Saved distance matrix {matrix.shape} → {path}")


# ─────────────────────────────────────────────────────────────────── #
#  Figure generators                                                   #
# ─────────────────────────────────────────────────────────────────── #

def plot_static_scene_3d(points: dict,
                          title: str = "Static Scene (3D)",
                          output_path: str = None,
                          max_points: int = 50000):
    """
    Paper Figure 17/20: 3D scatter of the static scene.
    White points on dark background matching paper style.
    """
    x = points['x']
    y = points['y']
    z = points['z']

    # Subsample for plotting if needed
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[idx], y[idx], z[idx]

    fig = plt.figure(figsize=(12, 9), facecolor='black')
    ax  = fig.add_subplot(111, projection='3d', facecolor='black')

    ax.scatter(x, y, z, c='white', s=0.18, alpha=0.9, linewidths=0)
    _style_scene_axis(ax, x, y, z)
    ax.set_xlabel('X (m)', color='white')
    ax.set_ylabel('Y (m)', color='white')
    ax.set_zlabel('Z (m)', color='white')
    ax.set_title(title, color='white', fontsize=13)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        logger.info(f"Saved 3D static scene plot → {output_path}")
    plt.close()


def plot_static_with_outliers(static_pts: dict,
                               outlier_pts: dict,
                               title: str = "Static Scene with Outliers",
                               output_path: str = None,
                               max_points: int = 50000):
    """
    Paper Figure 17: static=white, outlier=red on black background.
    """
    fig = plt.figure(figsize=(12, 9), facecolor='black')
    ax  = fig.add_subplot(111, projection='3d', facecolor='black')

    def _sub(pts, n):
        x, y, z = pts['x'], pts['y'], pts['z']
        if len(x) > n:
            idx = np.random.choice(len(x), n, replace=False)
            return x[idx], y[idx], z[idx]
        return x, y, z

    sx, sy, sz = _sub(static_pts,  max_points)
    ox, oy, oz = _sub(outlier_pts, max_points // 5)

    ax.scatter(sx, sy, sz, c='white', s=0.18, alpha=0.9, linewidths=0,
               label=f'Static ({len(static_pts["x"]):,})')
    if len(ox):
        ax.scatter(ox, oy, oz, c='red', s=1.2, alpha=0.95, linewidths=0,
                   label=f'Outlier ({len(outlier_pts["x"]):,})')

    _style_scene_axis(ax, sx, sy, sz)
    ax.set_xlabel('X (m)', color='white')
    ax.set_ylabel('Y (m)', color='white')
    ax.set_zlabel('Z (m)', color='white')
    ax.set_title(title, color='white', fontsize=13)
    leg = ax.legend(loc='upper right', fontsize=9)
    for text in leg.get_texts():
        text.set_color('white')
    leg.get_frame().set_facecolor('black')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='black')
        logger.info(f"Saved static+outlier plot → {output_path}")
    plt.close()


def plot_static_scene_topdown(points: dict,
                              title: str = "Static Scene (Top-down)",
                              output_path: str = None,
                              max_points: int = 70000):
    """Readable top-down projection with height encoded by color."""
    x = points['x']
    y = points['y']
    z = points['z']

    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.set_facecolor('black')
    sc = ax.scatter(x, y, c=z, s=0.22, cmap='gray', alpha=0.9, linewidths=0)
    ax.set_xlabel('X (m)', color='white')
    ax.set_ylabel('Y (m)', color='white')
    ax.set_title(title, color='white', fontsize=13)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.grid(False)
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label('Z (m)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(*_axis_limits(x))
    ax.set_ylim(*_axis_limits(y))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        logger.info(f"Saved top-down static scene plot → {output_path}")
    plt.close()


def plot_element_distance_distribution(dists: np.ndarray,
                                        labels: np.ndarray,
                                        channel: int,
                                        azimuth_bin: int,
                                        alpha_res: float,
                                        best_eps: float,
                                        title_prefix: str = "",
                                        output_path: str = None):
    """
    Paper Figures 12 & 14: Distance vs Frame Number scatter.
    Coloured by cluster label.
    """
    n = len(dists)
    frame_nums = np.arange(n)

    unique_labels = sorted(set(labels))
    colours = {-1: 'gray'}
    cmap = plt.cm.get_cmap('tab10', max(1, len(unique_labels)))
    ci = 0
    for lb in unique_labels:
        if lb != -1:
            colours[lb] = cmap(ci)
            ci += 1

    az_deg = azimuth_bin * alpha_res

    fig, ax = plt.subplots(figsize=(9, 5))
    for lb in unique_labels:
        mask = labels == lb
        lbl = 'Outlier' if lb == -1 else f'Cluster {lb}'
        clr = colours[lb]
        ax.scatter(dists[mask], frame_nums[mask], s=4, c=[clr],
                   label=lbl, alpha=0.7)

    ax.set_xlabel('Distance (m)', fontsize=11)
    ax.set_ylabel('Frame Number', fontsize=11)
    az_str = f"{az_deg:.1f}°"
    full_title = (f"{title_prefix}Distance vs Frame Number\n"
                  f"Channel={channel}, Azimuth={az_str}, "
                  f"best_eps={best_eps:.3f}")
    ax.set_title(full_title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_dbscan_1d(dists: np.ndarray,
                   labels: np.ndarray,
                   channel: int,
                   azimuth_bin: int,
                   alpha_res: float,
                   best_eps: float,
                   min_pts: int,
                   silhouette: float = None,
                   intra_dist: float = None,
                   output_path: str = None):
    """
    Paper Figures 15 & 16: DBSCAN clustering output — 1D scatter.
    """
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    az_deg = azimuth_bin * alpha_res

    fig, ax = plt.subplots(figsize=(9, 4))

    cmap = plt.cm.get_cmap('tab10', max(1, n_clusters))
    ci = 0
    for lb in unique_labels:
        mask = labels == lb
        if lb == -1:
            ax.scatter(dists[mask], np.zeros(mask.sum()),
                       c='gray', s=20, marker='x', label='Outlier', zorder=5)
        else:
            ax.scatter(dists[mask], np.zeros(mask.sum()),
                       c=[cmap(ci)], s=20, label=f'Cluster {lb}', alpha=0.8)
            ci += 1

    ax.set_xlabel('Distance (m)', fontsize=11)
    ax.set_yticks([])

    score_str = ""
    if silhouette is not None:
        score_str += f"Silhouette Score: {silhouette:.2f}, "
    if intra_dist is not None:
        score_str += f"Intra-cluster Distance: {intra_dist:.2f}"

    title = (f"Best EPS: {best_eps:.4f}, Min Samples: {min_pts}\n"
             + score_str + "\n"
             f"Channel={channel}, Azimuth={az_deg:.1f}°")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_distance_matrix_heatmap(static_matrix: np.ndarray,
                                  output_path: str = None):
    """Visualise the [C, M] static distance matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(14, 4))
    C, M = static_matrix.shape
    im = ax.imshow(static_matrix, aspect='auto', cmap='plasma',
                   vmin=0, vmax=np.percentile(static_matrix[static_matrix > 0], 95)
                   if np.any(static_matrix > 0) else 1.0)
    ax.set_xlabel('Azimuth Bin (m → 0°..360°)', fontsize=11)
    ax.set_ylabel('Laser Channel', fontsize=11)
    ax.set_title('Static Distance Matrix (Elementwise)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Distance (m)')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        logger.info(f"Saved heatmap → {output_path}")
    plt.close()


def plot_silhouette_sweep(channel: int,
                           azimuth_bin: int,
                           alpha_res: float,
                           eps_vals: np.ndarray,
                           sil_scores: np.ndarray,
                           output_path: str = None):
    """
    Paper Figure 10: Silhouette Scores for different eps / MinPts.
    """
    az_deg = azimuth_bin * alpha_res
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps_vals, sil_scores, 'b-o', markersize=4)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('eps', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title(f'Silhouette Scores — Channel={channel}, Azimuth={az_deg:.1f}°',
                 fontsize=10)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────── #
#  Summary report                                                      #
# ─────────────────────────────────────────────────────────────────── #

def save_stats_report(stats: dict, cfg, output_path: str):
    """Write a plain-text summary matching what the paper reports."""
    lines = [
        "=" * 60,
        "  STATIC SCENE CONSTRUCTION — RESULTS SUMMARY",
        "  (Rajput et al., ISPRS 2024 — Paper Replication)",
        "=" * 60,
        f"",
        f"Sensor:              {cfg.sensor_name}",
        f"Channels (C):        {cfg.num_channels}",
        f"Azimuth res (α_res): {cfg.azimuth_resolution:.3f}°",
        f"Azimuth bins (M):    {cfg.num_azimuth_bins}",
        f"Total elements (C×M):{cfg.num_elements:,}",
        f"",
        f"Frames processed:    {stats.get('n_frames', '?')}",
        f"",
        f"Clustering results:",
        f"  Static elements:       {stats['n_static']:,}  "
          f"({stats['coverage_pct']:.1f}%)",
        f"  Multi-cluster (trees): {stats['n_multi_cluster']:,}",
        f"  No-data elements:      {stats['n_nodata']:,}",
        f"  Outlier elements:      {stats['n_outlier']:,}",
        f"",
        "=" * 60,
    ]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved stats report → {output_path}")
    print('\n'.join(lines))
