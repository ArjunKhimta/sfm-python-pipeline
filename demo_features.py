"""Step 1 demo: detect SIFT features in two tractor photos and visualize matches.

Run from project root:
    python3 demo_features.py
    python3 demo_features.py --img1 samples/0002.jpg --img2 samples/0003.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from sfm.features import detect_features, draw_matches, match_features
from sfm.io_utils import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SIFT matches between two images")
    parser.add_argument("--img1", default="samples/0000.jpg")
    parser.add_argument("--img2", default="samples/0001.jpg")
    parser.add_argument("--max-dim", type=int, default=1024,
                        help="Resize so longest side == max_dim (speeds up SIFT)")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Lowe's ratio test threshold")
    parser.add_argument("--output", default="output/match_visualization.png")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.img1} and {args.img2} (resizing longest side to {args.max_dim}px)...")
    img1 = load_image(args.img1, max_dim=args.max_dim)
    img2 = load_image(args.img2, max_dim=args.max_dim)
    print(f"  img1 shape: {img1.shape},  img2 shape: {img2.shape}")

    print("Detecting SIFT features...")
    feat1 = detect_features(img1)
    feat2 = detect_features(img2)
    print(f"  img1: {len(feat1.keypoints)} keypoints")
    print(f"  img2: {len(feat2.keypoints)} keypoints")

    print(f"Matching with Lowe's ratio test (ratio={args.ratio})...")
    matches = match_features(feat1, feat2, ratio=args.ratio)
    print(f"  {len(matches)} good matches survived the ratio test")

    if len(matches) == 0:
        print("No matches — try a different image pair or relax the ratio threshold.")
        return

    print(f"Rendering visualization to {output_path}...")
    vis = draw_matches(img1, feat1, img2, feat2, matches, max_draw=80)
    cv2.imwrite(str(output_path), vis)
    print(f"Done. Open {output_path} to inspect the matches.")


if __name__ == "__main__":
    main()
