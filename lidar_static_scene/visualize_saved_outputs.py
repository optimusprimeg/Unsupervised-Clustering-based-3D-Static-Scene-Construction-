"""
visualize_saved_outputs.py
--------------------------
Regenerate visualizations from saved pipeline outputs without rerunning
frame loading or clustering.

Expected inputs in the output directory:
  - static_scene.pcd
  - outlier_points.pcd
  - static_distance_matrix.npy

The script writes paper-style figures to the chosen output directory.
"""

import argparse
import logging
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from src.pcd_reader import read_pcd, normalise_cloud
from src.dbscan_clustering import cluster_element
from src.visualizer import (
    plot_static_scene_3d,
    plot_static_scene_topdown,
    plot_static_with_outliers,
    plot_element_distance_distribution,
    plot_dbscan_1d,
    plot_distance_matrix_heatmap,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('visualize_saved_outputs')


def _load_point_cloud(path: str) -> dict:
    """Load a PCD file and return the normalised point dict."""
    cloud = read_pcd(path)
    return normalise_cloud(cloud)


def _parse_results_summary(path: str) -> dict:
    """Extract the main sensor settings from results_summary.txt if present."""
    data = {}
    if not os.path.exists(path):
        return data

    with open(path, 'r') as f:
        text = f.read()

    patterns = {
        'num_channels': r'Channels \(C\):\s+(\d+)',
        'azimuth_resolution': r'Azimuth res \(α_res\):\s+([0-9.]+)°',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            data[key] = float(match.group(1)) if key == 'azimuth_resolution' else int(match.group(1))
    return data


def _parse_example_elements(items: list[str]) -> list[tuple[int, int]]:
    """Parse CLI values in the form channel,azimuth_bin."""
    pairs = []
    for item in items:
        parts = item.split(',')
        if len(parts) != 2:
            raise ValueError(f'Invalid example element "{item}"; expected channel,azimuth_bin')
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def _cluster_element_from_agg(agg: np.ndarray, channel: int, azimuth_bin: int, min_pts_floor: int = 100):
    """Run DBSCAN element clustering for one (channel, azimuth_bin)."""
    if channel < 0 or channel >= agg.shape[0] or azimuth_bin < 0 or azimuth_bin >= agg.shape[1]:
        return None

    dists = agg[channel, azimuth_bin, :]
    dists = dists[dists > 0.0]
    if len(dists) < 2:
        return None

    return cluster_element(
        dists,
        channel=channel,
        azimuth_bin=azimuth_bin,
        eps_initial=0.08,
        eps_max=0.40,
        eps_step=0.01,
        min_pts_fraction=0.01,
        min_pts_floor=min_pts_floor,
    )


def _find_nearby_valid_result(agg: np.ndarray,
                              channel: int,
                              azimuth_bin: int,
                              max_radius: int = 120,
                              min_pts_floor: int = 100):
    """Find a nearby element that can be clustered when requested one is sparse."""
    result = _cluster_element_from_agg(agg, channel, azimuth_bin, min_pts_floor=min_pts_floor)
    if result is not None:
        return result

    for radius in range(1, max_radius + 1):
        for dm in (-radius, radius):
            m2 = azimuth_bin + dm
            if 0 <= m2 < agg.shape[1]:
                result = _cluster_element_from_agg(agg, channel, m2, min_pts_floor=min_pts_floor)
                if result is not None:
                    return result
    return None


def _find_multi_cluster_examples(agg: np.ndarray,
                                 max_examples: int = 2,
                                 max_checks: int = 300,
                                 min_pts_floor: int = 100) -> list:
    """Find a few elements that produce more than one DBSCAN cluster."""
    found = []
    c_step = max(1, agg.shape[0] // 16)
    m_step = max(1, agg.shape[1] // 250)

    checks = 0

    # Try random candidates first for a better chance of hitting multi-cluster elements.
    rng = np.random.default_rng(42)
    random_checks = min(max_checks // 2, 200)
    for _ in range(random_checks):
        c = int(rng.integers(0, agg.shape[0]))
        m = int(rng.integers(0, agg.shape[1]))
        result = _cluster_element_from_agg(agg, c, m, min_pts_floor=min_pts_floor)
        checks += 1
        if result is None:
            continue
        if result.n_clusters > 1:
            found.append(result)
            if len(found) >= max_examples:
                return found

    for c in range(0, agg.shape[0], c_step):
        for m in range(0, agg.shape[1], m_step):
            if checks >= max_checks:
                return found
            result = _cluster_element_from_agg(agg, c, m, min_pts_floor=min_pts_floor)
            checks += 1
            if result is None:
                continue
            if result.n_clusters > 1:
                found.append(result)
                if len(found) >= max_examples:
                    return found
    return found


def _plot_occlusion_single_frame(static_pts: dict, outlier_pts: dict, output_path: str):
    """Create a paper-like Figure 19 style occlusion view from saved outputs."""
    if len(static_pts['x']) == 0:
        return

    sx, sy = static_pts['x'], static_pts['y']
    ox = outlier_pts.get('x', np.array([]))
    oy = outlier_pts.get('y', np.array([]))

    if len(sx) > 90000:
        idx = np.random.choice(len(sx), 90000, replace=False)
        sx, sy = sx[idx], sy[idx]
    if len(ox) > 12000:
        idx = np.random.choice(len(ox), 12000, replace=False)
        ox, oy = ox[idx], oy[idx]

    fig, ax = plt.subplots(figsize=(12, 6.2), facecolor='black')
    ax.set_facecolor('black')
    ax.scatter(sx, sy, s=0.16, c='white', alpha=0.92, linewidths=0)
    if len(ox) > 0:
        ax.scatter(ox, oy, s=2.5, c='red', alpha=0.85, linewidths=0)

    ax.set_title('Figure 19. Occlusion in Single Frame (Saved Outputs)', color='white', fontsize=12)
    ax.set_xlabel('X (m)', color='white')
    ax.set_ylabel('Y (m)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.set_aspect('equal', adjustable='box')

    if len(ox) > 20:
        x0, y0 = float(np.median(ox)), float(np.median(oy))
        dx, dy = 6.0, 6.0
        ax.add_patch(plt.Rectangle((x0 - dx / 2, y0 - dy / 2), dx, dy,
                                   edgecolor='yellow', facecolor='none', linewidth=1.1))

        iax = inset_axes(ax, width='34%', height='46%', loc='upper right', borderpad=1.2)
        iax.set_facecolor('black')
        iax.scatter(sx, sy, s=0.13, c='white', alpha=0.7, linewidths=0)
        iax.scatter(ox, oy, s=3.2, c='red', alpha=0.9, linewidths=0)
        iax.set_xlim(x0 - dx / 2, x0 + dx / 2)
        iax.set_ylim(y0 - dy / 2, y0 + dy / 2)
        iax.tick_params(colors='white', labelsize=7)
        for spine in iax.spines.values():
            spine.set_color('yellow')

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight', facecolor='black')
    plt.close()


def _plot_silhouette_combinations(dists: np.ndarray,
                                  channel: int,
                                  azimuth_bin: int,
                                  output_path: str):
    """Create Figure-10 style silhouette plot over (min_pts, eps) combinations."""
    if len(dists) < 5:
        return

    X = dists.reshape(-1, 1)

    # Paper-like paired combinations for visualization.
    min_pts_values = [100, 110, 120, 130, 140, 150, 160, 170]
    eps_values = [0.08, 0.40]

    # If element has far fewer points than paper setup, fixed 100..170
    # collapses to one value (n-1). Switch to adaptive min_pts values.
    effective_fixed = [min(v, len(dists) - 1) for v in min_pts_values]
    if len(set(effective_fixed)) <= 2:
        low = max(5, int(0.15 * len(dists)))
        high = max(low + 1, int(0.85 * len(dists)))
        adaptive = np.linspace(low, high, 8)
        min_pts_values = sorted(set(int(v) for v in adaptive))

    labels_txt = []
    sil_vals = []

    for mpts in min_pts_values:
        for eps in eps_values:
            labels_txt.append(f"({mpts},{eps:.2f})")
            min_samples = min(max(2, mpts), len(dists) - 1)
            db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree')
            labels = db.fit_predict(X)
            non_noise = labels != -1
            unique_non_noise = set(labels[non_noise])
            if non_noise.sum() < 2 or len(unique_non_noise) < 2:
                sil_vals.append(-1.0)
                continue
            try:
                sil = silhouette_score(X[non_noise], labels[non_noise])
                sil_vals.append(float(sil))
            except Exception:
                sil_vals.append(-1.0)

    x = np.arange(len(labels_txt))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, sil_vals, color='blue', linewidth=1.3, marker='o', markersize=4,
            markerfacecolor='red', markeredgecolor='red')
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_xlabel('Combination of (min_points, eps)', fontsize=11)
    ax.set_title('Silhouette Scores for Different Combinations of min points and eps', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_txt, rotation=90)
    ax.grid(True, alpha=0.35)

    # Add element id in a compact annotation for traceability.
    ax.text(0.01, 0.02, f'Element ({channel},{azimuth_bin})', transform=ax.transAxes,
            fontsize=9, va='bottom', ha='left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close()


def regenerate_visuals(input_dir: str,
                      output_dir: str,
                      example_elements: list[tuple[int, int]] = None,
                      auto_multi_examples: int = 2) -> None:
    """Regenerate saved-output visualizations from existing artifacts."""
    static_scene_path = os.path.join(input_dir, 'static_scene.pcd')
    outlier_path = os.path.join(input_dir, 'outlier_points.pcd')
    matrix_path = os.path.join(input_dir, 'static_distance_matrix.npy')
    summary_path = os.path.join(input_dir, 'results_summary.txt')

    if not os.path.exists(static_scene_path):
        raise FileNotFoundError(f'Missing: {static_scene_path}')
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f'Missing: {matrix_path}')

    os.makedirs(output_dir, exist_ok=True)

    logger.info('Loading saved static scene from %s', static_scene_path)
    static_pts = _load_point_cloud(static_scene_path)

    outlier_pts = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
    if os.path.exists(outlier_path):
        logger.info('Loading saved outliers from %s', outlier_path)
        outlier_pts = _load_point_cloud(outlier_path)
    else:
        logger.warning('Outlier file not found: %s', outlier_path)

    logger.info('Loading static distance matrix from %s', matrix_path)
    static_matrix = np.load(matrix_path)

    sensor_info = _parse_results_summary(summary_path)
    alpha_res = sensor_info.get('azimuth_resolution', 0.2)

    example_dir = os.path.join(output_dir, 'examples')
    os.makedirs(example_dir, exist_ok=True)

    aggregated_path = os.path.join(input_dir, 'aggregated_matrix.npy')
    if not os.path.exists(aggregated_path):
        aggregated_path = os.path.join('lidar_static_scene', 'output', 'aggregated_matrix.npy')
    if not os.path.exists(aggregated_path):
        aggregated_path = os.path.join('output', 'aggregated_matrix.npy')

    agg = None
    if os.path.exists(aggregated_path):
        logger.info('Loading aggregated matrix from %s', aggregated_path)
        agg = np.load(aggregated_path)
    else:
        logger.warning('aggregated_matrix.npy not found; skipping DBSCAN example figures')

    if example_elements and agg is not None:
        logger.info('Generating %d requested DBSCAN example figures from saved aggregated data', len(example_elements))
        for channel, azimuth_bin in example_elements:
            result = _find_nearby_valid_result(agg, channel, azimuth_bin)
            if result is None:
                logger.warning('Skipping (%d,%d) because clustering was not possible', channel, azimuth_bin)
                continue

            if result.channel != channel or result.azimuth_bin != azimuth_bin:
                logger.info('Requested (%d,%d) used nearby valid element (%d,%d)',
                            channel, azimuth_bin, result.channel, result.azimuth_bin)

            tag = f'ch{channel}_az{azimuth_bin}'
            plot_element_distance_distribution(
                result.distances,
                result.labels,
                result.channel,
                result.azimuth_bin,
                alpha_res,
                result.best_eps,
                title_prefix='Multi Cluster Distribution of Aggregated Distance Element ',
                output_path=os.path.join(example_dir, f'fig12_{tag}_dist_scatter.png')
            )
            plot_dbscan_1d(
                result.distances,
                result.labels,
                result.channel,
                result.azimuth_bin,
                alpha_res,
                result.best_eps,
                result.best_min_pts,
                silhouette=result.silhouette,
                intra_dist=result.intra_dist,
                output_path=os.path.join(example_dir, f'fig15_{tag}_dbscan.png')
            )
            _plot_silhouette_combinations(
                result.distances,
                result.channel,
                result.azimuth_bin,
                output_path=os.path.join(example_dir, f'fig10_{tag}_silhouette_combinations.png')
            )

    if agg is not None and auto_multi_examples > 0:
        auto_results = _find_multi_cluster_examples(agg, max_examples=auto_multi_examples)
        relaxed_mode = False
        if len(auto_results) == 0:
            logger.info('No multi-cluster examples found with MinPts floor=100. Trying visualization-only relaxed floor=20.')
            auto_results = _find_multi_cluster_examples(
                agg,
                max_examples=auto_multi_examples,
                min_pts_floor=20,
            )
            relaxed_mode = len(auto_results) > 0
        logger.info('Found %d auto multi-cluster example(s)', len(auto_results))
        for i, result in enumerate(auto_results, start=1):
            tag = f'auto{i}_ch{result.channel}_az{result.azimuth_bin}'
            prefix = 'Multi Cluster Distribution of Aggregated Distance Element '
            if relaxed_mode:
                prefix = '[Viz-relaxed MinPts] Multi Cluster Distribution of Aggregated Distance Element '
            plot_element_distance_distribution(
                result.distances,
                result.labels,
                result.channel,
                result.azimuth_bin,
                alpha_res,
                result.best_eps,
                title_prefix=prefix,
                output_path=os.path.join(example_dir, f'fig12_{tag}_dist_scatter.png')
            )
            plot_dbscan_1d(
                result.distances,
                result.labels,
                result.channel,
                result.azimuth_bin,
                alpha_res,
                result.best_eps,
                result.best_min_pts,
                silhouette=result.silhouette,
                intra_dist=result.intra_dist,
                output_path=os.path.join(example_dir, f'fig15_{tag}_dbscan.png')
            )
            _plot_silhouette_combinations(
                result.distances,
                result.channel,
                result.azimuth_bin,
                output_path=os.path.join(example_dir, f'fig10_{tag}_silhouette_combinations.png')
            )

    plot_static_scene_3d(
        static_pts,
        title='Figure 20. Combined Frames Static Scene',
        output_path=os.path.join(output_dir, 'fig20_combined_frames_static_scene.png')
    )

    plot_static_scene_topdown(
        static_pts,
        title='Static Scene (Top-down) - From Saved Outputs',
        output_path=os.path.join(output_dir, 'saved_static_scene_topdown.png')
    )

    if len(outlier_pts['x']) > 0:
        plot_static_with_outliers(
            static_pts,
            outlier_pts,
            title='Static Scene with Outliers - From Saved Outputs',
            output_path=os.path.join(output_dir, 'saved_static_with_outliers.png')
        )
        _plot_occlusion_single_frame(
            static_pts,
            outlier_pts,
            output_path=os.path.join(output_dir, 'fig19_occlusion_single_frame.png')
        )

    plot_distance_matrix_heatmap(
        static_matrix,
        output_path=os.path.join(output_dir, 'saved_static_distance_matrix_heatmap.png')
    )

    logger.info('Saved regenerated figures to %s', output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Regenerate visualizations from saved pipeline outputs.'
    )
    parser.add_argument('--input_dir', default='output/', help='Directory with saved outputs')
    parser.add_argument('--output_dir', default='output/visuals/', help='Directory for regenerated figures')
    parser.add_argument('--example_elements', nargs='*', default=['10,36', '0,36'],
                        help='Selected elements as channel,azimuth_bin pairs (default: 10,36 0,36)')
    parser.add_argument('--auto_multi_examples', type=int, default=2,
                        help='Number of auto-discovered DBSCAN examples with >1 clusters')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    regenerate_visuals(
        args.input_dir,
        args.output_dir,
        _parse_example_elements(args.example_elements),
        auto_multi_examples=args.auto_multi_examples,
    )