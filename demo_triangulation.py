"""Step 3 demo: end-to-end on one image pair → first 3D point cloud (.ply).

Pipeline:
  features → match → RANSAC F → essential matrix → recover pose →
  DLT triangulation → cheirality + reprojection-error filtering →
  sample colors from img1 → write PLY.

Run from project root:
    python3 demo_triangulation.py
    python3 demo_triangulation.py --img1 images/0010.jpg --img2 images/0011.jpg

Open the result:
    open output/pair_cloud.ply           # uses MeshLab if installed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sfm.features import detect_features, match_features, matched_points
from sfm.geometry import ransac_essential, recover_pose
from sfm.io_utils import estimate_intrinsics, load_image, write_ply
from sfm.triangulation import (
    build_projection,
    points_in_front_of,
    reprojection_error,
    sample_colors,
    triangulate_dlt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", default="images/0000.jpg")
    parser.add_argument("--img2", default="images/0001.jpg")
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=1.0)
    parser.add_argument("--reproj-thresh", type=float, default=4.0,
                        help="Drop points with reprojection error > this (pixels)")
    parser.add_argument("--output", default="output/pair_cloud.ply")
    args = parser.parse_args()

    out_path = Path(args.output)

    # --- Load + features + matches ---
    print(f"Loading {args.img1} and {args.img2}...")
    img1 = load_image(args.img1, max_dim=args.max_dim)
    img2 = load_image(args.img2, max_dim=args.max_dim)

    print("Detecting + matching features...")
    feat1 = detect_features(img1)
    feat2 = detect_features(img2)
    matches = match_features(feat1, feat2, ratio=args.ratio)
    pts1, pts2 = matched_points(feat1, feat2, matches)
    print(f"  {len(matches)} matches passed Lowe's ratio test")

    # --- Essential matrix via 5-point + RANSAC ---
    # We use cv2.findEssentialMat here (not our F → E path) because going
    # through F is over-parameterized and triangulation accuracy degrades.
    # See the long comment in sfm/geometry.py. The from-scratch 8-point
    # implementation is showcased in demo_geometry.py instead.
    K = estimate_intrinsics(img1.shape)
    print(f"Fitting E via cv2.findEssentialMat (5-point + RANSAC, "
          f"threshold={args.ransac_thresh})...")
    res = ransac_essential(pts1, pts2, K, threshold=args.ransac_thresh)
    print(f"  Inliers: {res.n_inliers} / {len(matches)}")

    inlier_pts1 = pts1[res.inlier_mask]
    inlier_pts2 = pts2[res.inlier_mask]

    # --- Pose recovery ---
    pose = recover_pose(res.E, inlier_pts1, inlier_pts2, K)
    print(f"  Cheirality-valid pose inliers: {pose.n_inliers} / {len(inlier_pts1)}")

    # --- Build the two camera projection matrices ---
    # Convention: camera 1 sits at the world origin (R = I, t = 0).
    # Camera 2's pose is what recover_pose just gave us.
    R1, t1 = np.eye(3), np.zeros(3)
    R2, t2 = pose.R, pose.t.ravel()
    P1 = build_projection(K, R1, t1)
    P2 = build_projection(K, R2, t2)

    # Triangulate only the cheirality-valid matches.
    keep = pose.inlier_mask
    pts1_ok = inlier_pts1[keep]
    pts2_ok = inlier_pts2[keep]

    # --- Triangulation: from-scratch DLT, validated against OpenCV ---
    print("Triangulating with from-scratch DLT...")
    points_3d = triangulate_dlt(P1, P2, pts1_ok, pts2_ok)

    # OpenCV reference (returns 4xN homogeneous; divide to get Euclidean).
    cv_h = cv2.triangulatePoints(P1, P2, pts1_ok.T, pts2_ok.T)
    cv_pts = (cv_h[:3] / cv_h[3]).T
    diffs = np.linalg.norm(points_3d - cv_pts, axis=1)
    print(f"  Triangulated {len(points_3d)} points")
    print(f"  Mean Euclidean distance to cv2.triangulatePoints output: "
          f"{diffs.mean():.6f}  (small = our DLT agrees)")

    # --- Filter: positive depth in BOTH cameras + reprojection error ---
    front1 = points_in_front_of(points_3d, R1, t1)
    front2 = points_in_front_of(points_3d, R2, t2)
    err1 = reprojection_error(points_3d, pts1_ok, P1)
    err2 = reprojection_error(points_3d, pts2_ok, P2)
    print(f"  Reprojection error percentiles (img1): "
          f"50th={np.percentile(err1, 50):.2f}px  "
          f"90th={np.percentile(err1, 90):.2f}px  "
          f"max={err1.max():.2f}px")
    print(f"  Reprojection error percentiles (img2): "
          f"50th={np.percentile(err2, 50):.2f}px  "
          f"90th={np.percentile(err2, 90):.2f}px  "
          f"max={err2.max():.2f}px")
    good = front1 & front2 & (err1 < args.reproj_thresh) & (err2 < args.reproj_thresh)

    print(f"  Survived cheirality + reprojection filter "
          f"(err < {args.reproj_thresh}px): {int(good.sum())} / {len(points_3d)}")
    print(f"  Mean reprojection error of survivors: "
          f"img1={err1[good].mean():.2f}px  img2={err2[good].mean():.2f}px")

    final_pts = points_3d[good]
    final_colors = sample_colors(img1, pts1_ok[good])

    # --- Write PLY ---
    write_ply(out_path, final_pts, final_colors)
    print(f"\nWrote {len(final_pts)} colored 3D points: {out_path}")
    print("Open it in MeshLab, CloudCompare, or drag-and-drop into "
          "https://3dviewer.net for an in-browser look.")


if __name__ == "__main__":
    main()
