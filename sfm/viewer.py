"""Interactive 3D viewer + screenshot rendering using PyVista.

PyVista (built on VTK) is used in place of Open3D because Open3D doesn't
yet ship Python 3.13 wheels. Same end result: an interactive window where
you can orbit the point cloud, plus offscreen-rendered screenshots for
the README.

Camera frustums are drawn as small wireframe pyramids pointing in each
camera's view direction. The point cloud is colored from the per-point
RGB sampled during triangulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def filter_outliers(points: np.ndarray, colors: np.ndarray,
                    percentile_lo: float = 2.0,
                    percentile_hi: float = 98.0) -> tuple[np.ndarray, np.ndarray]:
    """Drop points outside the [lo, hi] percentile range on any axis.

    Triangulation occasionally produces points very far from the scene
    (numerically unstable rays). These outliers blow up the bounding box,
    making the actual cloud invisible at the rendered scale. Clipping by
    percentile is a robust, parameter-free way to keep the viewable cloud.
    """
    if len(points) == 0:
        return points, colors
    lo = np.percentile(points, percentile_lo, axis=0)
    hi = np.percentile(points, percentile_hi, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep], colors[keep]


def _camera_frustum(R: np.ndarray, t: np.ndarray, scale: float) -> pv.PolyData:
    """Build a small wireframe pyramid representing one camera pose.

    Camera convention: x_cam = R @ x_world + t, so the camera center in
    world coords is C = -R^T t. The image plane sits at depth +1 (in
    camera coords), shrunk to `scale` in world units. Five vertices:
    apex at the camera center + 4 corners on the image plane.
    """
    R_T = R.T
    C = -R_T @ t.reshape(3)

    # 4 corners on a unit-depth image plane (in camera coords).
    aspect = 1.0
    near = scale
    corners_cam = np.array([
        [-aspect, -1.0, 1.0],
        [ aspect, -1.0, 1.0],
        [ aspect,  1.0, 1.0],
        [-aspect,  1.0, 1.0],
    ]) * near

    # World coords: x_world = R^T (x_cam - t). For points already given in
    # camera coords as offsets from the camera center, just rotate by R^T.
    corners_world = (R_T @ corners_cam.T).T + C

    points = np.vstack([C[None, :], corners_world])  # 5 x 3
    # Lines: apex to each of the 4 corners, plus the 4 edges of the rectangle.
    lines = np.array([
        2, 0, 1,
        2, 0, 2,
        2, 0, 3,
        2, 0, 4,
        2, 1, 2,
        2, 2, 3,
        2, 3, 4,
        2, 4, 1,
    ])
    pd = pv.PolyData(points)
    pd.lines = lines
    return pd


def _scene_scale(points: np.ndarray) -> float:
    """Pick a frustum size that's small relative to the point cloud's extent."""
    if len(points) == 0:
        return 0.05
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    return float(extent * 0.03)


def _build_plotter(points: np.ndarray,
                   colors: np.ndarray,
                   cameras: dict[int, tuple[np.ndarray, np.ndarray]],
                   off_screen: bool = False,
                   point_size: float = 6.0) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=off_screen, window_size=(1200, 900))
    plotter.set_background("white")

    if len(points) > 0:
        cloud = pv.PolyData(points)
        # PyVista wants RGB as float [0,1] OR a (N,3) uint8 array tagged as
        # rgb=True. We stay with uint8.
        cloud["RGB"] = colors
        plotter.add_mesh(cloud, scalars="RGB", rgb=True,
                         point_size=point_size, render_points_as_spheres=True)

    scale = _scene_scale(points)
    for R, t in cameras.values():
        plotter.add_mesh(_camera_frustum(R, t, scale),
                         color="red", line_width=1.5)

    plotter.add_axes()
    return plotter


def show_reconstruction(points: np.ndarray,
                        colors: np.ndarray,
                        cameras: dict[int, tuple[np.ndarray, np.ndarray]],
                        point_size: float = 6.0) -> None:
    """Open an interactive PyVista window: orbit / zoom / pan the cloud."""
    plotter = _build_plotter(points, colors, cameras,
                             off_screen=False, point_size=point_size)
    print("\nViewer controls:")
    print("  Left-drag  -> rotate")
    print("  Right-drag -> zoom")
    print("  Middle-drag (or shift+drag) -> pan")
    print("  q / esc    -> close")
    plotter.show()


def save_screenshots(points: np.ndarray,
                     colors: np.ndarray,
                     cameras: dict[int, tuple[np.ndarray, np.ndarray]],
                     out_dir: str | Path,
                     point_size: float = 6.0) -> list[Path]:
    """Render 4 standard views to PNG files: front, side, top, perspective.

    Uses PyVista's offscreen renderer so it works without a display.
    Returns the list of written PNG paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(points) == 0:
        print("No points to render.")
        return []

    center = points.mean(axis=0)
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    distance = extent * 1.5

    views = {
        "front":       (center + np.array([0, 0, -distance]), center, (0, -1, 0)),
        "side":        (center + np.array([distance, 0, 0]),  center, (0, -1, 0)),
        "top":         (center + np.array([0, -distance, 0]), center, (0, 0, 1)),
        "perspective": (center + np.array([distance * 0.7, -distance * 0.5, -distance * 0.7]),
                        center, (0, -1, 0)),
    }

    written = []
    for name, (cam_pos, focal, up) in views.items():
        plotter = _build_plotter(points, colors, cameras,
                                 off_screen=True, point_size=point_size)
        plotter.camera_position = [tuple(cam_pos), tuple(focal), tuple(up)]
        out_path = out_dir / f"{name}.png"
        plotter.screenshot(str(out_path))
        plotter.close()
        written.append(out_path)
        print(f"  wrote {out_path}")
    return written
