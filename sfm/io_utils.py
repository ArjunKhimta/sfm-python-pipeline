"""Image loading and basic preprocessing for the SfM pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(folder: str | Path) -> list[Path]:
    """Return sorted image paths in `folder`. Errors clearly if none found."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")

    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return paths


def load_image(path: str | Path, max_dim: int | None = 1024) -> np.ndarray:
    """Load a BGR image, optionally downscaling so the longest side == max_dim.

    Downscaling speeds SIFT up dramatically without losing useful features.
    Pass max_dim=None to keep original resolution.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")

    if max_dim is not None:
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            new_size = (int(round(w * scale)), int(round(h * scale)))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    return img


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write an ASCII PLY of (N, 3) points with optional (N, 3) uint8 RGB colors.

    ASCII PLY is verbose but bulletproof — opens cleanly in MeshLab,
    CloudCompare, Open3D, and any web viewer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(points)
    has_color = colors is not None
    if has_color and len(colors) != n:
        raise ValueError("colors must match points length")

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        if has_color:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def estimate_intrinsics(image_shape: tuple[int, int, int],
                        focal_ratio: float = 1.2) -> np.ndarray:
    """Build a rough camera intrinsics matrix K when calibration is unknown.

    Assumes a centered principal point and equal focal length in x/y.
    `focal_ratio` * max(width, height) is a reasonable starting estimate
    for a typical phone/DSLR; we'll refine this later via bundle adjustment.
    """
    h, w = image_shape[:2]
    f = focal_ratio * max(h, w)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1.0]])
    return K
