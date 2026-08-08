"""Step 2 demo: estimate fundamental matrix, essential matrix, and camera pose
between two tractor photos. Compares the from-scratch 8-point + RANSAC
implementation against cv2.findFundamentalMat as a sanity check.

Run from project root:
    python3 demo_geometry.py
    python3 demo_geometry.py --img1 images/0010.jpg --img2 images/0011.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sfm.features import detect_features, draw_matches, match_features, matched_points
from sfm.geometry import (
    essential_from_fundamental,
    ransac_fundamental,
    recover_pose,
)
from sfm.io_utils import estimate_intrinsics, load_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", default="images/0000.jpg")
    parser.add_argument("--img2", default="images/0001.jpg")
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=1.0,
                        help="Sampson distance inlier threshold (pixel^2-ish)")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.img1} and {args.img2}...")
    img1 = load_image(args.img1, max_dim=args.max_dim)
    img2 = load_image(args.img2, max_dim=args.max_dim)

    print("Detecting + matching features...")
    feat1 = detect_features(img1)
    feat2 = detect_features(img2)
    matches = match_features(feat1, feat2, ratio=args.ratio)
    pts1, pts2 = matched_points(feat1, feat2, matches)
    print(f"  {len(matches)} matches passed Lowe's ratio test")

    if len(matches) < 30:
        print("Too few matches to fit a reliable F. Try a closer image pair.")
        return

    # ----- Our from-scratch RANSAC + 8-point -----
    print(f"\nFitting F via from-scratch RANSAC + 8-point (threshold={args.ransac_thresh})...")
    result = ransac_fundamental(pts1, pts2, threshold=args.ransac_thresh)
    print(f"  RANSAC iterations:       {result.iterations}")
    print(f"  Inliers:                 {result.n_inliers} / {len(matches)} "
          f"({100 * result.n_inliers / len(matches):.1f}%)")
    print("  F (ours, normalized):")
    print(result.F / np.linalg.norm(result.F))

    # ----- OpenCV reference -----
    F_cv, mask_cv = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, args.ransac_thresh, 0.999
    )
    cv_inliers = int(mask_cv.sum())
    print(f"\ncv2.findFundamentalMat reference:")
    print(f"  Inliers:                 {cv_inliers} / {len(matches)} "
          f"({100 * cv_inliers / len(matches):.1f}%)")
    print("  F (OpenCV, normalized):")
    print(F_cv / np.linalg.norm(F_cv))

    # Sanity: how close are the two F matrices? Compare normalized.
    F_ours_n = result.F / np.linalg.norm(result.F)
    F_cv_n = F_cv / np.linalg.norm(F_cv)
    # F is defined up to sign; align signs before measuring distance.
    if np.sign(F_ours_n.flat[np.argmax(np.abs(F_ours_n))]) != \
       np.sign(F_cv_n.flat[np.argmax(np.abs(F_cv_n))]):
        F_cv_n = -F_cv_n
    diff = np.linalg.norm(F_ours_n - F_cv_n)
    print(f"  Frobenius distance (ours vs OpenCV): {diff:.4f}  "
          f"(small = our implementation agrees)")

    # ----- Essential matrix + pose recovery -----
    K = estimate_intrinsics(img1.shape)
    print(f"\nIntrinsics K (estimated, focal_ratio=1.2):")
    print(K)

    inlier_pts1 = pts1[result.inlier_mask]
    inlier_pts2 = pts2[result.inlier_mask]

    E = essential_from_fundamental(result.F, K, K)
    pose = recover_pose(E, inlier_pts1, inlier_pts2, K)
    print(f"\nRecovered camera pose (cam1 -> cam2):")
    print("  R:")
    print(pose.R)
    print(f"  t (unit length, direction only): {pose.t.ravel()}")
    print(f"  Cheirality-valid points: {pose.n_inliers} / {len(inlier_pts1)}")

    # ----- Save inlier-only match visualization -----
    inlier_matches = matches[result.inlier_mask]
    vis = draw_matches(img1, feat1, img2, feat2, inlier_matches, max_draw=80)
    out_path = out / "geometry_inliers.png"
    cv2.imwrite(str(out_path), vis)
    print(f"\nSaved inlier-only matches: {out_path}")
    print("(Compare against output/match_visualization.png — you should see "
          "the wild outlier lines from before are gone.)")


if __name__ == "__main__":
    main()
