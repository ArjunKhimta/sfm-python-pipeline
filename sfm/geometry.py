"""Two-view epipolar geometry: from-scratch fundamental matrix + RANSAC + pose.

Implements the normalized 8-point algorithm (Hartley) and a RANSAC loop
using Sampson distance for inlier scoring. Essential matrix is derived
from F via E = K2^T F K1, and pose recovery (R, t) uses cv2.recoverPose
because it bundles the cheirality check (positive depth in both views),
which is bookkeeping rather than geometry.

The 8-point algorithm itself is implemented from scratch and validated
against cv2.findFundamentalMat in the geometry demo.

Reference: Hartley & Zisserman, "Multiple View Geometry in Computer Vision",
chapters 9 (epipolar geometry) and 11 (F estimation).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Normalized 8-point algorithm
# ---------------------------------------------------------------------------

def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: translate centroid to origin, scale so mean
    distance from origin == sqrt(2). Returns (normalized_pts, T) where T
    is the 3x3 similarity transform applied (in homogeneous coordinates).

    Why: solving the 8-point system on raw pixel coordinates is numerically
    awful because pixel values span 0..1000s. Normalization makes the
    constraint matrix well-conditioned. This is the single most important
    fix in Hartley's classic 1997 paper.
    """
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    if mean_dist < 1e-12:
        raise ValueError("Degenerate point set (all points coincide)")
    scale = np.sqrt(2.0) / mean_dist

    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1.0]])

    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_norm = (T @ pts_h.T).T[:, :2]
    return pts_norm, T


def eight_point_F(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Estimate the fundamental matrix from >=8 correspondences.

    Implements Hartley's normalized 8-point algorithm:
      1. Normalize both point sets.
      2. Build constraint matrix A from x2^T F x1 = 0 (one row per match).
      3. Solve via SVD: F is the right singular vector for the smallest
         singular value, reshaped to 3x3.
      4. Enforce rank-2 constraint (F is rank-deficient by construction;
         noise breaks this, so we project back via SVD with sigma_3 := 0).
      5. Denormalize: F = T2^T * F_norm * T1.
    """
    if len(pts1) < 8 or len(pts2) < 8:
        raise ValueError(f"Need >=8 points, got {len(pts1)}")

    p1, T1 = _normalize_points(pts1)
    p2, T2 = _normalize_points(pts2)

    # Build constraint matrix: each match gives one row of A.
    # x2^T F x1 = 0 expands to 9 terms in F's entries.
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    A = np.column_stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1,      y1,      np.ones_like(x1),
    ])

    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # Enforce rank 2 (the algebraically true F is rank-deficient).
    U, S, Vt2 = np.linalg.svd(F_norm)
    S[-1] = 0.0
    F_norm = U @ np.diag(S) @ Vt2

    # Denormalize.
    F = T2.T @ F_norm @ T1
    return F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------

def _sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """First-order geometric error approximation per match.

    Better than the algebraic error |x2^T F x1| because it accounts for the
    expected reprojection error in the image plane. Standard inlier
    criterion in modern F/E estimation pipelines.
    """
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    Fx1 = (F @ pts1_h.T).T          # (N, 3)
    Ftx2 = (F.T @ pts2_h.T).T       # (N, 3)
    x2tFx1 = np.sum(pts2_h * Fx1, axis=1)  # (N,)

    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return (x2tFx1 ** 2) / np.maximum(denom, 1e-12)


@dataclass
class FundamentalRansacResult:
    F: np.ndarray
    inlier_mask: np.ndarray  # bool array, length == len(matches)
    iterations: int

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum())


def ransac_fundamental(pts1: np.ndarray,
                       pts2: np.ndarray,
                       threshold: float = 1.0,
                       confidence: float = 0.999,
                       max_iters: int = 5000,
                       seed: int = 0) -> FundamentalRansacResult:
    """RANSAC loop around the 8-point algorithm.

    `threshold` is the Sampson distance (in pixels^2 roughly) below which a
    match is an inlier. `confidence` adapts the iteration count: we keep
    running until the probability of having sampled an all-inlier minimal
    set is >= confidence.

    Final step refits F on ALL inliers (not just the minimal sample) for
    better accuracy — this is "Gold Standard" practice.
    """
    if len(pts1) != len(pts2):
        raise ValueError("pts1 and pts2 must have the same length")
    if len(pts1) < 8:
        raise ValueError(f"Need >=8 matches, got {len(pts1)}")

    rng = np.random.default_rng(seed)
    n = len(pts1)
    best_F = None
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0

    iters = max_iters
    i = 0
    while i < iters:
        sample_idx = rng.choice(n, 8, replace=False)
        try:
            F_candidate = eight_point_F(pts1[sample_idx], pts2[sample_idx])
        except (np.linalg.LinAlgError, ValueError):
            i += 1
            continue

        errors = _sampson_distance(F_candidate, pts1, pts2)
        inliers = errors < threshold
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_F = F_candidate

            # Adaptive iteration count: how many trials do we need to be
            # `confidence` sure we've sampled at least one all-inlier set?
            inlier_ratio = count / n
            if inlier_ratio > 0:
                denom = np.log(max(1.0 - inlier_ratio ** 8, 1e-12))
                needed = int(np.ceil(np.log(1.0 - confidence) / denom))
                iters = min(iters, max(needed, 1))

        i += 1

    if best_F is None or best_count < 8:
        raise RuntimeError(f"RANSAC failed to find a usable F (best inliers: {best_count})")

    # Gold-standard refinement: refit on all inliers.
    best_F = eight_point_F(pts1[best_inliers], pts2[best_inliers])
    return FundamentalRansacResult(F=best_F, inlier_mask=best_inliers, iterations=i)


# ---------------------------------------------------------------------------
# Essential matrix and pose recovery
# ---------------------------------------------------------------------------
#
# Two paths exist below:
#
#   essential_from_fundamental(F, K1, K2)
#     Used in demo_geometry.py to *compare* our from-scratch 8-point F to
#     OpenCV's. Going F -> E is algebraically over-parameterized (F has 7
#     DoF, E has 5) so the resulting E carries extra noise and triangulated
#     points end up with ~5-10px reprojection error. Fine for showing the
#     math, not fine for the actual reconstruction.
#
#   ransac_essential(pts1, pts2, K)
#     Wraps cv2.findEssentialMat (Nister's 5-point algorithm) which fits E
#     directly under the correct constraints. Used by the reconstruction
#     pipeline. Sub-pixel reprojection error in practice.
#
# Implementing the 5-point algorithm from scratch is ~hundreds of lines of
# polynomial root finding, well beyond the scope of this learning project.

def essential_from_fundamental(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """E = K2^T F K1 (Hartley-Zisserman eq 9.12).

    Then enforce E's two equal non-zero singular values (rank 2, with
    sigma_1 == sigma_2), which is E's defining algebraic property.
    """
    E = K2.T @ F @ K1
    U, _, Vt = np.linalg.svd(E)
    # Force singular values to (1, 1, 0) — E is defined up to scale.
    E_clean = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    # Keep right-handed coordinate system (det should stay > 0 across U,Vt).
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E_clean = -E_clean
    return E_clean


@dataclass
class EssentialRansacResult:
    E: np.ndarray
    inlier_mask: np.ndarray  # bool array, length == len(input matches)

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum())


def ransac_essential(pts1: np.ndarray,
                     pts2: np.ndarray,
                     K: np.ndarray,
                     threshold: float = 1.0,
                     confidence: float = 0.999) -> EssentialRansacResult:
    """Pipeline-grade E estimation via OpenCV's 5-point + RANSAC.

    Used by the reconstruction pipeline because it gives sub-pixel
    reprojection errors after triangulation. See the long comment above
    for why we don't use the from-scratch F → E path here.
    """
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, confidence, threshold)
    if E is None:
        raise RuntimeError("cv2.findEssentialMat failed")
    return EssentialRansacResult(E=E, inlier_mask=mask.ravel().astype(bool))


@dataclass
class PoseResult:
    R: np.ndarray   # 3x3 rotation: from cam1 frame to cam2 frame
    t: np.ndarray   # 3x1 translation (unit length — scale is unobservable from 2 views)
    inlier_mask: np.ndarray  # bool array filtered by cheirality (positive depth)

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum())


def recover_pose(E: np.ndarray,
                 pts1: np.ndarray,
                 pts2: np.ndarray,
                 K: np.ndarray) -> PoseResult:
    """Decompose E into (R, t) and pick the physically valid solution.

    E has 4 algebraic decompositions; only one places triangulated points
    *in front of* both cameras (the cheirality constraint). cv2.recoverPose
    runs the cheirality check internally, which would be ~30 lines of
    bookkeeping to reimplement — we use it directly and document why.
    """
    n_in, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return PoseResult(R=R, t=t, inlier_mask=mask.ravel().astype(bool))
