"""Feature detection and matching.

We use OpenCV's SIFT for keypoint detection (scale + rotation invariant,
robust against the specular surfaces in our tractor dataset). Matching uses
brute-force L2 with Lowe's ratio test to filter ambiguous matches.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ImageFeatures:
    """SIFT keypoints + descriptors for one image."""
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray  # shape (N, 128), float32

    @property
    def points(self) -> np.ndarray:
        """Keypoint pixel coordinates as (N, 2) float32 array."""
        return np.array([kp.pt for kp in self.keypoints], dtype=np.float32)


def detect_features(image: np.ndarray,
                    n_features: int = 0,
                    contrast_threshold: float = 0.04) -> ImageFeatures:
    """Detect SIFT keypoints and compute descriptors.

    n_features=0 means "as many as SIFT finds" — we don't cap by default
    because dropping features costs us potential triangulations later.
    Lower contrast_threshold = more (but weaker) features.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features,
                           contrastThreshold=contrast_threshold)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError("SIFT found no features — image may be blank or corrupt")

    return ImageFeatures(keypoints=tuple(keypoints), descriptors=descriptors)


def match_features(feat1: ImageFeatures,
                   feat2: ImageFeatures,
                   ratio: float = 0.75) -> np.ndarray:
    """Match descriptors with Lowe's ratio test, returns (M, 2) index pairs.

    Each row is [idx_in_feat1, idx_in_feat2]. We use BFMatcher with
    knnMatch(k=2) and keep matches where the best is < `ratio` * second-best
    — Lowe's classic test for filtering out ambiguous matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(feat1.descriptors, feat2.descriptors, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))

    return np.array(good, dtype=np.int32) if good else np.zeros((0, 2), dtype=np.int32)


def matched_points(feat1: ImageFeatures,
                   feat2: ImageFeatures,
                   matches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert match index pairs into the actual (x, y) coordinates in each image."""
    pts1 = feat1.points[matches[:, 0]]
    pts2 = feat2.points[matches[:, 1]]
    return pts1, pts2


def draw_matches(img1: np.ndarray, feat1: ImageFeatures,
                 img2: np.ndarray, feat2: ImageFeatures,
                 matches: np.ndarray,
                 max_draw: int = 80) -> np.ndarray:
    """Render a side-by-side visualization of matches for debugging.

    Drawing all matches for a tractor pair (~thousands) is unreadable, so
    we randomly subsample down to `max_draw` for the visualization.
    """
    if len(matches) > max_draw:
        idx = np.random.default_rng(0).choice(len(matches), max_draw, replace=False)
        sample = matches[idx]
    else:
        sample = matches

    dmatches = [cv2.DMatch(int(a), int(b), 0) for a, b in sample]
    return cv2.drawMatches(
        img1, feat1.keypoints,
        img2, feat2.keypoints,
        dmatches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
