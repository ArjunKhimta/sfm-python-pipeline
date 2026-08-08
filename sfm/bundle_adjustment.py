"""Sparse bundle adjustment with scipy.optimize.least_squares.

Bundle adjustment jointly refines all camera poses and 3D point positions
to minimize the total reprojection error across every observation. It's
the standard SfM finishing step -- without it, small errors at each
incremental PnP step compound across the chain ("drift").

Parameters being optimized:
  - 6 numbers per camera (3 Rodrigues angles + 3 translation)
  - 3 numbers per 3D point
  - Camera 0 is FIXED at the origin to remove gauge ambiguity (otherwise
    BA could trivially rotate/translate the whole scene without changing
    the cost). This is standard practice.

Cost: sum over (camera_idx, point_idx, observed_pixel_xy) of
      || project(camera, point) - observed_pixel ||^2

The Jacobian is sparse -- residual i only depends on the parameters of
its observation's camera and point, not on every other camera/point. We
build a `lil_matrix` sparsity pattern so least_squares uses sparse linear
algebra. This makes BA tractable for our 30 cameras + ~10k points.

Reference: scipy cookbook ("Large-scale bundle adjustment in scipy"),
Hartley & Zisserman chapter 18 ("N-View Computational Methods").
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


@dataclass
class BAObservations:
    """Flat representation of all observations to feed least_squares."""
    camera_ids: np.ndarray  # (N_obs,) int -- index into cameras_order
    point_ids: np.ndarray   # (N_obs,) int -- index into points
    pixels: np.ndarray      # (N_obs, 2) float -- observed (x, y)


def _flatten_observations(rec) -> tuple[BAObservations, list[int]]:
    """Walk the reconstruction's observations, drop ones we can't use,
    and renumber camera ids 0..M-1 in registration order.

    Returns the flat observations + the camera order (list of original
    image indices) so we can map BA outputs back to rec.cameras.
    """
    cameras_order = sorted(rec.cameras.keys())
    cam_id_for_image = {img_idx: i for i, img_idx in enumerate(cameras_order)}

    cam_ids, pt_ids, pix = [], [], []
    for image_idx, feat_to_pt in rec.observations.items():
        if image_idx not in cam_id_for_image:
            continue
        if image_idx not in rec.feature_pixels:
            continue
        pixels_this_image = rec.feature_pixels[image_idx]
        cam_id = cam_id_for_image[image_idx]
        for feat_idx, pt_idx in feat_to_pt.items():
            cam_ids.append(cam_id)
            pt_ids.append(pt_idx)
            pix.append(pixels_this_image[feat_idx])

    return BAObservations(
        camera_ids=np.array(cam_ids, dtype=np.int32),
        point_ids=np.array(pt_ids, dtype=np.int32),
        pixels=np.array(pix, dtype=np.float64),
    ), cameras_order


def _pack_params(rec, cameras_order: list[int]) -> np.ndarray:
    """Encode current state as a single 1D parameter vector.

    Layout: [cam_1_rvec, cam_1_t, cam_2_rvec, cam_2_t, ..., points.flatten()]
    Camera 0 is NOT included -- it's the fixed reference frame.
    """
    params = []
    for img_idx in cameras_order[1:]:  # skip camera 0
        R, t = rec.cameras[img_idx]
        rvec, _ = cv2.Rodrigues(R)
        params.append(rvec.ravel())
        params.append(t.ravel())
    params.append(rec.points_array().ravel())
    return np.concatenate(params)


def _unpack_params(x: np.ndarray, n_cams: int, n_points: int):
    """Inverse of _pack_params. Returns (rvecs[n_cams-1], ts[n_cams-1], points[n_points])."""
    n_cam_params = 6 * (n_cams - 1)
    cam_part = x[:n_cam_params].reshape(n_cams - 1, 6)
    rvecs = cam_part[:, :3]
    ts = cam_part[:, 3:]
    points = x[n_cam_params:].reshape(n_points, 3)
    return rvecs, ts, points


def _project(points_world: np.ndarray, rvec: np.ndarray, t: np.ndarray,
             K: np.ndarray) -> np.ndarray:
    """Project (N, 3) world points through one camera. (N, 2) pixel coords."""
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    cam = (R @ points_world.T).T + t                    # (N, 3)
    proj_h = (K @ cam.T).T                              # (N, 3)
    return proj_h[:, :2] / proj_h[:, 2:3]


def _residuals(x: np.ndarray, K: np.ndarray, n_cams: int,
               obs: BAObservations) -> np.ndarray:
    """Stacked (N_obs * 2,) residual vector: projected pixel - observed pixel."""
    n_points = (len(x) - 6 * (n_cams - 1)) // 3
    rvecs, ts, points = _unpack_params(x, n_cams, n_points)

    # Camera 0 is the fixed reference: identity R, zero t.
    # Build per-observation projections by grouping by camera (avoids one
    # cv2.Rodrigues per observation).
    out = np.empty((len(obs.camera_ids), 2))
    for cam_id in range(n_cams):
        mask = obs.camera_ids == cam_id
        if not mask.any():
            continue
        pts = points[obs.point_ids[mask]]
        if cam_id == 0:
            rvec = np.zeros(3); t = np.zeros(3)
        else:
            rvec = rvecs[cam_id - 1]
            t = ts[cam_id - 1]
        out[mask] = _project(pts, rvec, t, K)
    return (out - obs.pixels).ravel()


def _build_sparsity(n_cams: int, n_points: int, obs: BAObservations) -> lil_matrix:
    """Each observation contributes 2 residual rows. Each row depends only
    on its observation's camera (6 params) and point (3 params)."""
    n_obs = len(obs.camera_ids)
    m = 2 * n_obs
    n = 6 * (n_cams - 1) + 3 * n_points
    A = lil_matrix((m, n), dtype=np.uint8)

    for i in range(n_obs):
        cam_id = obs.camera_ids[i]
        pt_id = obs.point_ids[i]
        # Camera params (skip camera 0 since it's not in the parameter vector).
        if cam_id != 0:
            cam_offset = 6 * (cam_id - 1)
            A[2 * i,     cam_offset:cam_offset + 6] = 1
            A[2 * i + 1, cam_offset:cam_offset + 6] = 1
        # Point params.
        pt_offset = 6 * (n_cams - 1) + 3 * pt_id
        A[2 * i,     pt_offset:pt_offset + 3] = 1
        A[2 * i + 1, pt_offset:pt_offset + 3] = 1
    return A


def _rmse(residuals: np.ndarray) -> float:
    """Root-mean-square reprojection error in pixels (per observation)."""
    n_obs = len(residuals) // 2
    per_obs = residuals.reshape(n_obs, 2)
    return float(np.sqrt((per_obs ** 2).sum(axis=1).mean()))


def run_bundle_adjustment(rec, max_iters: int = 50, verbose: bool = True):
    """Refine `rec` in place via sparse Levenberg-Marquardt-ish optimization.

    Returns (rmse_before, rmse_after, n_iterations). Camera 0's pose stays
    fixed; everything else moves to minimize total reprojection error.
    """
    obs, cameras_order = _flatten_observations(rec)
    n_cams = len(cameras_order)
    n_points = len(rec.points_3d)
    if verbose:
        print(f"  observations: {len(obs.camera_ids)}  cameras: {n_cams}  points: {n_points}")
        print(f"  free parameters: {6 * (n_cams - 1) + 3 * n_points}")

    x0 = _pack_params(rec, cameras_order)
    rmse_before = _rmse(_residuals(x0, rec.K, n_cams, obs))
    if verbose:
        print(f"  reprojection RMSE before BA: {rmse_before:.4f} px")

    sparsity = _build_sparsity(n_cams, n_points, obs)

    t0 = time.time()
    result = least_squares(
        _residuals, x0,
        jac_sparsity=sparsity,
        method="trf",
        x_scale="jac",
        ftol=1e-4,
        max_nfev=max_iters,
        verbose=2 if verbose else 0,
        args=(rec.K, n_cams, obs),
    )
    elapsed = time.time() - t0

    rmse_after = _rmse(result.fun)
    if verbose:
        print(f"  reprojection RMSE after  BA: {rmse_after:.4f} px  "
              f"({result.nfev} fn evals, {elapsed:.1f}s)")

    # Unpack and write back into rec.
    rvecs, ts, points = _unpack_params(result.x, n_cams, n_points)
    for i, img_idx in enumerate(cameras_order):
        if i == 0:
            rec.cameras[img_idx] = (np.eye(3), np.zeros(3))
        else:
            R, _ = cv2.Rodrigues(rvecs[i - 1].reshape(3, 1))
            rec.cameras[img_idx] = (R, ts[i - 1].copy())
    rec.points_3d = [p.copy() for p in points]

    return rmse_before, rmse_after, result.nfev
