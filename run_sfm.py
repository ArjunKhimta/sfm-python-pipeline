"""From-scratch SfM driver: turn `images/` into `output/tractor_sparse.ply`.

This is the pipeline counterpart to run_colmap.py (which uses COLMAP as a
black box). Here we use our own SIFT matching, RANSAC essential-matrix +
pose recovery, DLT triangulation, and incremental PnP-based registration,
optionally finished with sparse bundle adjustment.

Run from project root:
    python3 run_sfm.py                              # reconstruct + write PLY
    python3 run_sfm.py --refine                     # also run bundle adjustment
    python3 run_sfm.py --screenshots                # also save 4-view PNGs
    python3 run_sfm.py --show                       # open interactive viewer
    python3 run_sfm.py --refine --screenshots --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sfm.bundle_adjustment import run_bundle_adjustment
from sfm.io_utils import (
    estimate_intrinsics,
    list_images,
    load_image,
    write_ply,
)
from sfm.pipeline import run_pipeline
from sfm.viewer import filter_outliers, save_screenshots, show_reconstruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--output", default="output/tractor_sparse.ply")
    parser.add_argument("--save-state", default="output/reconstruction.pkl",
                        help="Pickle the full Reconstruction here (cameras + observations)")
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=1.0)
    parser.add_argument("--reproj-thresh", type=float, default=4.0)
    parser.add_argument("--refine", action="store_true",
                        help="Run bundle adjustment after the incremental chain")
    parser.add_argument("--ba-iters", type=int, default=50)
    parser.add_argument("--screenshots", action="store_true",
                        help="Render 4 standard-view PNGs to output/views/")
    parser.add_argument("--show", action="store_true",
                        help="Open the interactive PyVista viewer at the end")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = list_images(args.image_dir)
    print(f"Found {len(paths)} images in {args.image_dir}")

    sample = load_image(paths[0], max_dim=args.max_dim)
    K = estimate_intrinsics(sample.shape)
    print(f"Estimated intrinsics K (focal_ratio=1.2 of max(H,W)):\n{K}\n")

    rec = run_pipeline(
        paths,
        K=K,
        max_dim=args.max_dim,
        ratio=args.ratio,
        ransac_thresh=args.ransac_thresh,
        reproj_thresh=args.reproj_thresh,
    )

    if args.refine:
        print(f"\nRunning sparse bundle adjustment (max {args.ba_iters} iterations)...")
        before, after, nfev = run_bundle_adjustment(rec, max_iters=args.ba_iters)
        print(f"  RMSE: {before:.4f} px -> {after:.4f} px  "
              f"({100 * (before - after) / before:.1f}% reduction)")

    raw_points = rec.points_array()
    raw_colors = rec.colors_array()
    points, colors = filter_outliers(raw_points, raw_colors)
    print(f"\nOutlier filter (1st-99th percentile per axis): "
          f"{len(raw_points)} -> {len(points)} points")
    write_ply(out_path, points, colors)
    print(f"Wrote {len(points)} colored 3D points to {out_path}")
    print(f"Cameras registered: {len(rec.cameras)} / {len(paths)}")

    if args.save_state:
        rec.save(args.save_state)
        print(f"Saved full reconstruction state to {args.save_state}")

    if args.screenshots:
        print(f"\nRendering 4-view screenshots to output/views/...")
        save_screenshots(points, colors, rec.cameras, out_dir="output/views")

    if args.show:
        show_reconstruction(points, colors, rec.cameras)


if __name__ == "__main__":
    main()
