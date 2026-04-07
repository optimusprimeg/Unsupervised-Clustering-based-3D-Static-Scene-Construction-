"""
main.py
-------
Main entry point for the static scene construction pipeline.

Usage:
    python main.py --config config.yaml
    python main.py --pcd_dir data/ --max_frames 500
    python main.py --pcd_dir data/ --demo    # run on synthetic data

Paper: "Unsupervised Clustering-based 3D Static Scene Construction
        Using LiDAR Channel and Azimuth Angle"
        Rajput et al., ISPRS Archives, 2024
"""

import os
import sys
import argparse
import logging
import time
import numpy as np

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.pcd_reader       import load_pcd_frames
from src.sensor_config    import auto_detect, SensorConfig
from src.frame_extractor  import build_aggregated_distance_matrix, get_element_distances
from src.dbscan_clustering import cluster_all_elements
from src.dbscan_parallel   import cluster_all_elements_parallel
from src.static_scene     import reconstruct_3d, extract_moving_objects
from src.visualizer       import (write_pcd, save_distance_matrix,
                                   plot_static_scene_3d,
                                   plot_static_with_outliers,
                                   plot_distance_matrix_heatmap,
                                   plot_element_distance_distribution,
                                   plot_dbscan_1d,
                                   save_stats_report)

# ── Logging setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('main')


# ================================================================== #
#  Synthetic demo data generator (for testing without real dataset)   #
# ================================================================== #

def generate_synthetic_frames(n_frames: int = 200,
                                n_channels: int = 16,
                                alpha_res: float = 0.2,
                                seed: int = 42) -> list:
    """
    Generate synthetic VLP-16-style frames for pipeline testing.

    Scene: static buildings at ~20m (channels 6-15) and
           ~8m (channels 0-5), plus a moving object at random distance.
    """
    rng = np.random.default_rng(seed)
    n_az = int(360 / alpha_res)   # 1800 for VLP-16

    # VLP-16 vertical angles
    vangles = np.linspace(-15, 15, n_channels)

    # Static background distances: channel × azimuth
    bg_dist = np.zeros((n_channels, n_az), dtype=np.float32)
    for c in range(n_channels):
        if c < 8:
            # Lower channels: ground + near objects (~3-15m)
            bg_dist[c, :] = rng.uniform(3.0, 15.0, n_az)
        else:
            # Upper channels: buildings (~15-40m)
            bg_dist[c, :] = rng.uniform(15.0, 40.0, n_az)

    frames = []
    for _ in range(n_frames):
        pts = {'x': [], 'y': [], 'z': [], 'intensity': [], 'ring': [], 'distance': []}

        for c in range(n_channels):
            omega = np.radians(vangles[c])
            for m in range(n_az):
                alpha = np.radians(m * alpha_res)

                # Static return with small Gaussian noise
                R_static = bg_dist[c, m]
                R = R_static + rng.normal(0, 0.05)   # ±5cm noise

                # Occasionally inject a moving object (10% of azimuth bins)
                if rng.random() < 0.10 and c in range(5, 10) and R_static > 7.0:
                    R = rng.uniform(5.0, R_static - 1.0)

                if R < 0.5:
                    continue   # no return

                x = R * np.cos(omega) * np.sin(alpha)
                y = R * np.cos(omega) * np.cos(alpha)
                z = R * np.sin(omega)

                pts['x'].append(x)
                pts['y'].append(y)
                pts['z'].append(z)
                pts['intensity'].append(rng.uniform(0, 255))
                pts['ring'].append(c)
                pts['distance'].append(R)

        for k in pts:
            pts[k] = np.array(pts[k], dtype=np.float32 if k != 'ring' else np.int32)

        frames.append(pts)

    logger.info(f"Generated {n_frames} synthetic frames, "
                f"~{len(frames[0]['x']):,} pts/frame")
    return frames


# ================================================================== #
#  Main pipeline                                                       #
# ================================================================== #

def run_pipeline(pcd_dir:        str,
                 output_dir:     str,
                 max_frames:     int   = None,
                 frame_step:     int   = 1,
                 forced_channels: int  = None,
                 forced_alpha_res: float = None,
                 eps_initial:    float = 0.08,
                 eps_max:        float = 0.40,
                 eps_step:       float = 0.01,
                 min_pts_frac:   float = 0.01,
                 min_pts_floor:  int   = 100,
                 range_min:      float = 0.1,
                 range_max:      float = 100.0,
                 compute_backend: str  = 'auto',
                 n_jobs:         int   = -1,
                 demo_mode:      bool  = False,
                 visualize:      bool  = True):

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ── STEP 1: Load frames ──────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1: Data Loading")
    logger.info("=" * 55)

    if demo_mode:
        logger.info("DEMO MODE: using synthetic data (no real .pcd needed)")
        frames = generate_synthetic_frames(n_frames=200)
        forced_channels  = forced_channels  or 16
        forced_alpha_res = forced_alpha_res or 0.2
    else:
        frames = load_pcd_frames(pcd_dir, max_frames=max_frames,
                                  step=frame_step)

    logger.info(f"Loaded {len(frames)} frames.")

    # ── STEP 2: Sensor auto-detection ───────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 2: Sensor Auto-Detection")
    logger.info("=" * 55)

    cfg = auto_detect(frames,
                      forced_channels=forced_channels,
                      forced_alpha_res=forced_alpha_res)
    logger.info(str(cfg))

    # ── STEP 3: Build aggregated distance matrix ─────────────────────
    logger.info("=" * 55)
    logger.info("STEP 3: Frame Extraction & Aggregation")
    logger.info("=" * 55)

    agg = build_aggregated_distance_matrix(frames, cfg)
    # agg shape: [C, M, N_frames]

    if True:   # save intermediate
        np.save(os.path.join(output_dir, 'aggregated_matrix.npy'), agg)
        logger.info(f"Saved aggregated matrix {agg.shape}")

    # ── STEP 4: DBSCAN clustering per element ────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 4: DBSCAN Clustering (per element)")
    logger.info("=" * 55)
    logger.info("This is the most time-intensive step.")
    logger.info(f"Elements to process: {cfg.num_elements:,}")

    backend = compute_backend.lower()
    if backend not in ('auto', 'cpu', 'gpu'):
        raise ValueError("compute_backend must be one of: auto, cpu, gpu")

    if backend == 'gpu':
        static_matrix, cluster_info, stats = cluster_all_elements(
            agg,
            eps_initial=eps_initial,
            eps_max=eps_max,
            eps_step=eps_step,
            min_pts_fraction=min_pts_frac,
            min_pts_floor=min_pts_floor,
            use_gpu=True,
        )
    elif backend == 'cpu':
        static_matrix, cluster_info, stats = cluster_all_elements_parallel(
            agg,
            eps_initial=eps_initial,
            eps_max=eps_max,
            eps_step=eps_step,
            min_pts_fraction=min_pts_frac,
            min_pts_floor=min_pts_floor,
            n_jobs=n_jobs,
        )
    else:
        try:
            static_matrix, cluster_info, stats = cluster_all_elements(
                agg,
                eps_initial=eps_initial,
                eps_max=eps_max,
                eps_step=eps_step,
                min_pts_fraction=min_pts_frac,
                min_pts_floor=min_pts_floor,
                use_gpu=True,
            )
        except Exception as exc:
            logger.warning("GPU auto mode failed (%s). Falling back to CPU parallel.", exc)
            static_matrix, cluster_info, stats = cluster_all_elements_parallel(
                agg,
                eps_initial=eps_initial,
                eps_max=eps_max,
                eps_step=eps_step,
                min_pts_fraction=min_pts_frac,
                min_pts_floor=min_pts_floor,
                n_jobs=n_jobs,
            )
    stats['n_frames'] = len(frames)

    save_distance_matrix(
        os.path.join(output_dir, 'static_distance_matrix.npy'),
        static_matrix
    )

    # ── STEP 5: 3D Reconstruction ────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 5: 3D Static Scene Reconstruction")
    logger.info("=" * 55)

    static_pts = reconstruct_3d(static_matrix, cfg,
                                 range_min=range_min, range_max=range_max)

    write_pcd(os.path.join(output_dir, 'static_scene.pcd'), static_pts,
              extra_fields=['distance', 'channel', 'azimuth_deg'])

    # ── STEP 6: Extract outlier points (single reference frame) ──────
    logger.info("=" * 55)
    logger.info("STEP 6: Outlier Extraction (reference frame)")
    logger.info("=" * 55)

    ref_frame = frames[len(frames) // 2]   # middle frame as reference
    outlier_pts = extract_moving_objects(ref_frame, static_matrix, cfg,
                                          tolerance=0.5)
    write_pcd(os.path.join(output_dir, 'outlier_points.pcd'), outlier_pts,
              extra_fields=['distance', 'channel'])

    # ── STEP 7: Visualizations ───────────────────────────────────────
    if visualize:
        logger.info("=" * 55)
        logger.info("STEP 7: Generating Visualizations")
        logger.info("=" * 55)

        # Figure 20: Combined static scene (paper style)
        plot_static_scene_3d(
            static_pts,
            title="Combined Frames Static Scene",
            output_path=os.path.join(output_dir, 'fig20_static_scene_combined.png')
        )

        # Figure 17+18: Static + outliers in reference frame
        plot_static_with_outliers(
            static_pts, outlier_pts,
            title="Static Scene with Outlier Points (Reference Frame)",
            output_path=os.path.join(output_dir, 'fig17_static_with_outliers.png')
        )

        # Distance matrix heatmap
        plot_distance_matrix_heatmap(
            static_matrix,
            output_path=os.path.join(output_dir, 'static_distance_matrix_heatmap.png')
        )

        # Paper Figure 12/15: Multi-cluster example (channel 10, azimuth 36)
        # Find a channel that actually had multi-cluster behavior
        _plot_example_elements(cluster_info, cfg, agg, output_dir)

    # ── STEP 8: Report ───────────────────────────────────────────────
    save_stats_report(stats, cfg, os.path.join(output_dir, 'results_summary.txt'))

    elapsed = time.time() - t0
    logger.info(f"\nPipeline complete in {elapsed:.1f}s")
    logger.info(f"Outputs in: {output_dir}/")

    return static_matrix, static_pts, stats


def _plot_example_elements(cluster_info, cfg, agg, output_dir):
    """Plot paper-style figures for interesting elements."""
    C, M = cluster_info.shape

    multi_examples  = []
    single_examples = []

    for c in range(min(C, 16)):
        for m in range(0, min(M, 1800), 36):   # sample every 36 bins (~7.2°)
            result = cluster_info[c, m]
            if result is None:
                continue
            if result.had_multi_cluster and len(multi_examples) < 2:
                multi_examples.append(result)
            elif not result.had_multi_cluster and len(single_examples) < 2:
                single_examples.append(result)
            if len(multi_examples) >= 2 and len(single_examples) >= 2:
                break
        if len(multi_examples) >= 2 and len(single_examples) >= 2:
            break

    for i, result in enumerate(multi_examples):
        tag = f"multi_ch{result.channel}_az{result.azimuth_bin}"
        plot_element_distance_distribution(
            result.distances, result.labels,
            result.channel, result.azimuth_bin,
            cfg.azimuth_resolution, result.best_eps,
            title_prefix="[Multi-cluster] ",
            output_path=os.path.join(output_dir, f'fig12_{tag}_dist_scatter.png')
        )
        plot_dbscan_1d(
            result.distances, result.labels,
            result.channel, result.azimuth_bin,
            cfg.azimuth_resolution, result.best_eps, result.best_min_pts,
            silhouette=result.silhouette, intra_dist=result.intra_dist,
            output_path=os.path.join(output_dir, f'fig15_{tag}_dbscan.png')
        )

    for i, result in enumerate(single_examples):
        tag = f"single_ch{result.channel}_az{result.azimuth_bin}"
        plot_element_distance_distribution(
            result.distances, result.labels,
            result.channel, result.azimuth_bin,
            cfg.azimuth_resolution, result.best_eps,
            title_prefix="[Single-cluster] ",
            output_path=os.path.join(output_dir, f'fig14_{tag}_dist_scatter.png')
        )
        plot_dbscan_1d(
            result.distances, result.labels,
            result.channel, result.azimuth_bin,
            cfg.azimuth_resolution, result.best_eps, result.best_min_pts,
            silhouette=result.silhouette, intra_dist=result.intra_dist,
            output_path=os.path.join(output_dir, f'fig16_{tag}_dbscan.png')
        )


# ================================================================== #
#  CLI                                                                 #
# ================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description='Static Scene Construction (Rajput et al., ISPRS 2024)'
    )
    p.add_argument('--pcd_dir',    default='data/',    help='Input PCD directory')
    p.add_argument('--output_dir', default='output/',  help='Output directory')
    p.add_argument('--max_frames', type=int, default=None,
                   help='Max frames to load (None=all)')
    p.add_argument('--frame_step', type=int, default=1,
                   help='Load every Nth frame (1=all)')
    p.add_argument('--channels',   type=int, default=None,
                   help='Force num channels (None=auto-detect)')
    p.add_argument('--alpha_res',  type=float, default=None,
                   help='Force azimuth resolution deg (None=auto-detect)')
    p.add_argument('--eps_initial',type=float, default=0.08,
                   help='DBSCAN eps start (paper: 0.08)')
    p.add_argument('--eps_max',    type=float, default=0.40,
                   help='DBSCAN eps max (paper: 0.40)')
    p.add_argument('--eps_step',   type=float, default=0.01,
                   help='DBSCAN eps step (paper: 0.01)')
    p.add_argument('--no_viz',     action='store_true',
                   help='Skip visualization')
    p.add_argument('--demo',       action='store_true',
                   help='Run on synthetic data (no real dataset needed)')
    p.add_argument('--compute_backend', choices=['auto', 'cpu', 'gpu'], default='auto',
                   help='Compute backend for clustering (default: auto)')
    p.add_argument('--n_jobs', type=int, default=-1,
                   help='CPU worker processes for parallel mode (-1=all cores)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        pcd_dir         = args.pcd_dir,
        output_dir      = args.output_dir,
        max_frames      = args.max_frames,
        frame_step      = args.frame_step,
        forced_channels = args.channels,
        forced_alpha_res= args.alpha_res,
        eps_initial     = args.eps_initial,
        eps_max         = args.eps_max,
        eps_step        = args.eps_step,
        compute_backend = args.compute_backend,
        n_jobs          = args.n_jobs,
        demo_mode       = args.demo,
        visualize       = not args.no_viz,
    )
