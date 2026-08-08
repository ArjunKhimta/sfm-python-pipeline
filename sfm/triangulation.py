"""Linear triangulation (DLT) — from 2D pixel correspondences to 3D points.

Given two camera projection matrices P1, P2 (each 3x4 = K [R | t]) and
matched pixel coordinates, recovers the 3D point that best explains both
observations.

Implementation is from scratch and validated against cv2.triangulatePoints
in the demo.

Reference: Hartley & Zisserman, "Multiple View Geometry", section 12.2
("Linear triangulation methods").
"""

from __future__ import annotations

import numpy as np


def build_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble the 3x4 projection matrix P = K [R | t]."""
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray,
                    pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Triangulate 3D points from two views via Direct Linear Transform.

    For each match (x1, x2) and projection matrices P1, P2, the relation
    x = P X (in homogeneous coords) gives, after eliminating the scale,
    two independent linear equations per view. Stacking the four equations
    from both views yields a 4x4 system A X = 0, solved via SVD: X is the
    right singular vector for the smallest singular value.

    Inputs:
      P1, P2: 3x4 projection matrices
      pts1, pts2: (N, 2) pixel coordinates of matched points
    Returns:
      (N, 3) triangulated 3D points (Euclidean, after homogeneous divide)
    """
    if len(pts1) != len(pts2):
        raise ValueError("pts1 and pts2 must be the same length")

    points_3d = np.empty((len(pts1), 3))
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        # 4x4 matrix from the two cross-product constraints, two rows each.
        A = np.vstack([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X_hom = Vt[-1]
        points_3d[i] = X_hom[:3] / X_hom[3]

    return points_3d


def project_points(points_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project (N, 3) 3D points through 3x4 P; return (N, 2) pixel coords."""
    homog = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    proj = (P @ homog.T).T   # (N, 3)
    return proj[:, :2] / proj[:, 2:3]


def reprojection_error(points_3d: np.ndarray,
                       pts_2d: np.ndarray,
                       P: np.ndarray) -> np.ndarray:
    """Per-point Euclidean pixel error between observed 2D and projected 3D."""
    projected = project_points(points_3d, P)
    return np.linalg.norm(projected - pts_2d, axis=1)


def points_in_front_of(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Boolean mask: True where point has positive depth in the given camera.

    Cheirality check: a triangulated point is only physically meaningful
    if it lies in front of (positive Z in) both cameras.
    """
    cam_coords = (R @ points_3d.T + t.reshape(3, 1)).T   # (N, 3)
    return cam_coords[:, 2] > 0


def sample_colors(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Look up BGR colors at floating-point pixel coordinates (nearest neighbor).

    Returns (N, 3) uint8 colors in RGB order so they're ready for PLY output.
    """
    h, w = image.shape[:2]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    bgr = image[ys, xs]
    return bgr[:, ::-1]   # BGR -> RGB
