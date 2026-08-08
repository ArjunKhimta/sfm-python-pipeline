"""From-scratch Structure-from-Motion pipeline.

A learning implementation of the classical SfM pipeline using OpenCV
primitives (SIFT, BFMatcher) and from-scratch geometry where possible
(RANSAC, DLT triangulation, bundle adjustment).

Companion to run_colmap.py, which serves as the production baseline.
"""

__version__ = "0.1.0"
