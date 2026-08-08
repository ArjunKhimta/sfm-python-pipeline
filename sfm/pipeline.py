"""Incremental Structure-from-Motion across many images.

Strategy (the simplest version of "real" incremental SfM):
  1. Initialize the world frame with the first image pair: triangulate
     matched features to seed the 3D point set, set camera 0 at the origin
     and camera 1 at the recovered pose.
  2. For each subsequent image i:
       a. Match its features against image i-1.
       b. For matches whose i-1 feature already has a 3D point, collect
          (3D point, 2D pixel in image i) pairs. Use solvePnPRansac to
          recover camera i's pose in the existing world frame -- this is
          how scale stays consistent across the whole sequence.
       c. For matches whose i-1 feature does NOT yet have a 3D point,
          triangulate using cameras i-1 and i and add the new point.
  3. Stop when no more images can be registered (PnP fails) or we run out.

Key data structure: `observations[image_idx][feature_idx] -> 3d_point_idx`
lets us answer "does this 2D feature already correspond to a known 3D
point?" in O(1), which is what makes the chaining work.

What this does NOT do (yet -- step 5):
  - Bundle adjustment (no global refinement; small errors drift across
    the chain).
  - Loop closure or matching to non-adjacent images.

These omissions are intentional -- this is the simplest correct
incremental SfM, kept readable. We add bundle adjustment in step 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sfm.features import ImageFeatures, detect_features, match_features, matched_points
from sfm.geometry import ransac_essential, recover_pose
from sfm.io_utils import load_image
from sfm.triangulation import (
    build_projection,
    points_in_front_of,
    project_points,
    sample_colors,
    triangulate_dlt,
)


@dataclass
class Reconstruction:
    """All state for an incremental reconstruction."""
    K: np.ndarray
    points_3d: list[np.ndarray] = field(default_factory=list)
    point_colors: list[np.ndarray] = field(default_factory=list)
    cameras: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # observations[image_idx][feature_idx] = point_3d_idx
    observations: dict[int, dict[int, int]] = field(default_factory=dict)
    # feature_pixels[image_idx] = (n_features, 2) array of (x, y) pixel coords.
    # Captured during the pipeline so bundle adjustment has 2D observations
    # without needing to re-run SIFT. Optional -- only needed for BA.
    feature_pixels: dict[int, np.ndarray] = field(default_factory=dict)

    def add_point(self, xyz: np.ndarray, rgb: np.ndarray) -> int:
        idx = len(self.points_3d)
        self.points_3d.append(xyz)
        self.point_colors.append(rgb)
        return idx

    def link(self, image_idx: int, feature_idx: int, point_idx: int) -> None:
        self.observations.setdefault(image_idx, {})[feature_idx] = point_idx

    def get_point_for_feature(self, image_idx: int, feature_idx: int) -> int | None:
        return self.observations.get(image_idx, {}).get(feature_idx)

    def points_array(self) -> np.ndarray:
        return np.array(self.points_3d) if self.points_3d else np.zeros((0, 3))

    def colors_array(self) -> np.ndarray:
        return np.array(self.point_colors, dtype=np.uint8) if self.point_colors else np.zeros((0, 3), np.uint8)

    def save(self, path: str | "Path") -> None:
        """Persist as a pickle. Convenient for hand-off between pipeline,
        viewer, and bundle adjustment so we don't recompute SIFT each run."""
        import pickle
        from pathlib import Path as _P
        p = _P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | "Path") -> "Reconstruction":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def _projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return build_projection(K, R, t.reshape(3))


def _filter_by_reprojection(points: np.ndarray,
                            pts2d_list: list[tuple[np.ndarray, np.ndarray]],
                            max_err: float) -> np.ndarray:
    """Bool mask: True for points whose reprojection error is below max_err in
    EVERY view that observed them. Each entry of pts2d_list is (P, observed_2d).
    """
    keep = np.ones(len(points), dtype=bool)
    for P, observed in pts2d_list:
        proj = project_points(points, P)
        err = np.linalg.norm(proj - observed, axis=1)
        keep &= err < max_err
    return keep


def initialize_pair(rec: Reconstruction,
                    img0: np.ndarray, feat0: ImageFeatures, idx0: int,
                    img1: np.ndarray, feat1: ImageFeatures, idx1: int,
                    ratio: float, ransac_thresh: float, reproj_thresh: float) -> int:
    """Bootstrap the reconstruction from the first image pair.

    Camera idx0 sits at the origin (R=I, t=0); camera idx1 gets the pose
    recovered from the essential matrix. Returns the number of 3D points
    triangulated.
    """
    matches = match_features(feat0, feat1, ratio=ratio)
    if len(matches) < 30:
        raise RuntimeError(f"Init pair has only {len(matches)} matches -- need >= 30")

    pts0, pts1 = matched_points(feat0, feat1, matches)
    res = ransac_essential(pts0, pts1, rec.K, threshold=ransac_thresh)
    inlier_idx = np.where(res.inlier_mask)[0]
    pts0_in = pts0[inlier_idx]
    pts1_in = pts1[inlier_idx]

    pose = recover_pose(res.E, pts0_in, pts1_in, rec.K)
    R0, t0 = np.eye(3), np.zeros(3)
    R1, t1 = pose.R, pose.t.ravel()

    P0 = _projection(rec.K, R0, t0)
    P1 = _projection(rec.K, R1, t1)

    cheir = pose.inlier_mask
    pts0_ok = pts0_in[cheir]
    pts1_ok = pts1_in[cheir]
    inlier_idx_ok = inlier_idx[cheir]

    points = triangulate_dlt(P0, P1, pts0_ok, pts1_ok)
    front0 = points_in_front_of(points, R0, t0)
    front1 = points_in_front_of(points, R1, t1)
    keep = front0 & front1 & _filter_by_reprojection(
        points, [(P0, pts0_ok), (P1, pts1_ok)], reproj_thresh
    )

    rec.cameras[idx0] = (R0, t0)
    rec.cameras[idx1] = (R1, t1)
    colors = sample_colors(img0, pts0_ok[keep])

    feat0_idxs = matches[inlier_idx_ok[keep], 0]
    feat1_idxs = matches[inlier_idx_ok[keep], 1]

    for xyz, rgb, f0i, f1i in zip(points[keep], colors, feat0_idxs, feat1_idxs):
        pid = rec.add_point(xyz, rgb)
        rec.link(idx0, int(f0i), pid)
        rec.link(idx1, int(f1i), pid)

    return int(keep.sum())


def add_image(rec: Reconstruction,
              prev_img: np.ndarray, prev_feat: ImageFeatures, prev_idx: int,
              new_img: np.ndarray, new_feat: ImageFeatures, new_idx: int,
              ratio: float, reproj_thresh: float) -> tuple[int, int]:
    """Register a new image into the existing reconstruction.

    Returns (n_pnp_inliers, n_new_points). Raises if PnP fails.
    """
    matches = match_features(prev_feat, new_feat, ratio=ratio)
    if len(matches) < 20:
        raise RuntimeError(f"Only {len(matches)} matches to image {prev_idx} -- skipping")

    # Split matches into "already triangulated" (-> PnP) and "new" (-> later triangulate).
    pnp_3d, pnp_2d, pnp_match_rows = [], [], []
    new_match_rows = []
    for row, (f_prev, f_new) in enumerate(matches):
        pid = rec.get_point_for_feature(prev_idx, int(f_prev))
        if pid is not None:
            pnp_3d.append(rec.points_3d[pid])
            pnp_2d.append(new_feat.keypoints[int(f_new)].pt)
            pnp_match_rows.append((row, pid))
        else:
            new_match_rows.append(row)

    if len(pnp_3d) < 6:
        raise RuntimeError(f"Only {len(pnp_3d)} 2D-3D matches -- PnP needs >= 6")

    pnp_3d_arr = np.array(pnp_3d, dtype=np.float32)
    pnp_2d_arr = np.array(pnp_2d, dtype=np.float32)

    # solvePnPRansac: gives camera pose in world frame (R, t such that
    # x_cam = R @ X_world + t for points X_world). dist coeffs = 0 since
    # we don't model lens distortion in this pipeline.
    ok, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
        pnp_3d_arr, pnp_2d_arr, rec.K, distCoeffs=None,
        reprojectionError=reproj_thresh, confidence=0.999,
        iterationsCount=200, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers_pnp is None or len(inliers_pnp) < 6:
        raise RuntimeError(f"PnP failed (ok={ok}, inliers={None if inliers_pnp is None else len(inliers_pnp)})")

    R_new, _ = cv2.Rodrigues(rvec)
    t_new = tvec.ravel()
    rec.cameras[new_idx] = (R_new, t_new)

    # Link the PnP-inlier observations to the existing 3D points.
    inlier_set = set(int(i) for i in inliers_pnp.ravel())
    for local_row_idx, (match_row, pid) in enumerate(pnp_match_rows):
        if local_row_idx in inlier_set:
            f_new = int(matches[match_row, 1])
            rec.link(new_idx, f_new, pid)

    # Triangulate previously-unseen matches using prev and new cameras.
    n_new_points = 0
    if new_match_rows:
        R_prev, t_prev = rec.cameras[prev_idx]
        P_prev = _projection(rec.K, R_prev, t_prev)
        P_new = _projection(rec.K, R_new, t_new)

        new_rows_arr = np.array(new_match_rows, dtype=np.int32)
        f_prev_idx = matches[new_rows_arr, 0]
        f_new_idx = matches[new_rows_arr, 1]
        pts_prev = prev_feat.points[f_prev_idx]
        pts_new = new_feat.points[f_new_idx]

        triangulated = triangulate_dlt(P_prev, P_new, pts_prev, pts_new)
        front_prev = points_in_front_of(triangulated, R_prev, t_prev)
        front_new = points_in_front_of(triangulated, R_new, t_new)
        good = front_prev & front_new & _filter_by_reprojection(
            triangulated, [(P_prev, pts_prev), (P_new, pts_new)], reproj_thresh
        )

        colors = sample_colors(new_img, pts_new[good])
        for xyz, rgb, fp, fn in zip(triangulated[good], colors,
                                     f_prev_idx[good], f_new_idx[good]):
            pid = rec.add_point(xyz, rgb)
            rec.link(prev_idx, int(fp), pid)
            rec.link(new_idx, int(fn), pid)
            n_new_points += 1

    return len(inlier_set), n_new_points


def run_pipeline(image_paths: list[Path],
                 K: np.ndarray,
                 max_dim: int = 1024,
                 ratio: float = 0.75,
                 ransac_thresh: float = 1.0,
                 reproj_thresh: float = 4.0,
                 verbose: bool = True) -> Reconstruction:
    """Run incremental SfM across an ordered list of images."""
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images")

    # Pre-load all images and detect features once. With 30 images at 1024px
    # this fits comfortably in memory and avoids repeated disk + SIFT cost.
    if verbose:
        print(f"Loading {len(image_paths)} images and detecting SIFT features...")
    images, features = [], []
    for p in tqdm(image_paths, disable=not verbose):
        img = load_image(p, max_dim=max_dim)
        images.append(img)
        features.append(detect_features(img))

    rec = Reconstruction(K=K)
    # Cache the keypoint pixel arrays per image so bundle adjustment has
    # the 2D observations without rerunning SIFT.
    for i, feat in enumerate(features):
        rec.feature_pixels[i] = feat.points

    if verbose:
        print(f"\nInitializing from pair (0, 1)...")
    n_init = initialize_pair(
        rec, images[0], features[0], 0, images[1], features[1], 1,
        ratio=ratio, ransac_thresh=ransac_thresh, reproj_thresh=reproj_thresh,
    )
    if verbose:
        print(f"  Pair init: {n_init} 3D points, 2 cameras registered")

    skipped = 0
    for i in range(2, len(image_paths)):
        try:
            n_pnp, n_new = add_image(
                rec, images[i - 1], features[i - 1], i - 1,
                images[i], features[i], i,
                ratio=ratio, reproj_thresh=reproj_thresh,
            )
            if verbose:
                print(f"  +img {i:2d}: PnP inliers={n_pnp:4d}  new pts={n_new:4d}  "
                      f"total pts={len(rec.points_3d)}")
        except RuntimeError as e:
            skipped += 1
            if verbose:
                print(f"  +img {i:2d}: SKIPPED ({e})")

    if verbose:
        print(f"\nDone: {len(rec.cameras)} cameras registered, "
              f"{len(rec.points_3d)} 3D points, {skipped} images skipped")
    return rec
