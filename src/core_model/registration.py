"""Rigid (rotation + translation) point-set registration via the Kabsch algorithm.

A rigid transform preserves every distance and angle between the points it's
applied to — it can reposition a scan, but it can't distort the anatomy
within it. That's what makes it safe to apply to structures (like bowel)
that weren't part of the landmark fit: since the transform can't deform
anything, the fitted structure's position *relative to the landmarks* is
unchanged, only its coordinates relative to the target scan's frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RigidTransform:
    """A rotation + translation: target ≈ rotation @ source + translation."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    reflection_detected: bool = False # true if best alignment was a mirror (singnaling potential mirror image upload)

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return points @ self.rotation.T + self.translation

    def rotation_angle_deg(self) -> float:
        """Magnitude of the rotation, as a single angle (axis-angle norm)."""
        cos_theta = np.clip((np.trace(self.rotation) - 1) / 2, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> RigidTransform:
    """Solve for the rigid transform mapping `source` points onto `target` points.

    `source` and `target` must be (N, 3) arrays of corresponding points
    (same anatomical landmark, same order), with N >= 3 and not all
    collinear — otherwise rotation about the line they define is
    undetermined. Uses the Kabsch algorithm (orthogonal Procrustes): the
    least-squares-optimal rigid transform, solved in closed form via SVD.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape:
        raise ValueError(
            f"source and target must have the same shape, got {source.shape} vs {target.shape}"
        )
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"expected (N, 3) arrays, got shape {source.shape}")
    if source.shape[0] < 3:
        raise ValueError("need at least 3 landmark pairs to determine rotation")

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    # The raw best-fit orthogonal matrix (before correction) can be reflection, correct it then signal
    d = np.sign(np.linalg.det(vt.T @ u.T))
    reflection_detected = bool(d < 0)
    if reflection_detected:
        logger.warning(
            "Best-fit alignment required a reflection, — check for a mirrored/flipped input"
            "orientation or laterality issue) before trusting this result."
        )
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    translation = target_centroid - rotation @ source_centroid

    return RigidTransform(
        rotation=rotation,
        translation=translation,
        reflection_detected=reflection_detected,
    )


def registration_rmse(
    transform: RigidTransform, source: np.ndarray, target: np.ndarray
) -> float:
    """Root-mean-square landmark error after applying `transform` to `source`."""
    aligned = transform.apply(source)
    target = np.asarray(target, dtype=float)
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
