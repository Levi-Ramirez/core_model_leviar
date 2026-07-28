import logging

import numpy as np
import pytest

from core_model.registration import fit_rigid_transform, registration_rmse

# a non-collinear, non-planar set of reference points to transform in tests
REFERENCE_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [5.0, 5.0, 5.0],
    ]
)


def rotation_about_z(degrees: float) -> np.ndarray:
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_pure_translation_is_recovered_exactly():
    translation = np.array([12.0, -4.0, 7.0])
    target = REFERENCE_POINTS + translation

    transform = fit_rigid_transform(REFERENCE_POINTS, target)

    assert np.allclose(transform.translation, translation, atol=1e-8)
    assert np.allclose(transform.rotation, np.eye(3), atol=1e-8)
    assert registration_rmse(transform, REFERENCE_POINTS, target) < 1e-8
    assert transform.reflection_detected is False


def test_rotation_and_translation_are_recovered_exactly():
    true_rotation = rotation_about_z(37.0)
    true_translation = np.array([3.0, -8.0, 15.0])
    target = REFERENCE_POINTS @ true_rotation.T + true_translation

    transform = fit_rigid_transform(REFERENCE_POINTS, target)

    assert np.allclose(transform.rotation, true_rotation, atol=1e-6)
    assert np.allclose(transform.translation, true_translation, atol=1e-6)
    assert transform.rotation_angle_deg() == pytest.approx(37.0, abs=1e-4)
    assert registration_rmse(transform, REFERENCE_POINTS, target) < 1e-6
    assert transform.reflection_detected is False


def test_returns_proper_rotation_even_when_landmarks_are_mirrored(caplog):
    """The raw, unconstrained least-squares fit (plain SVD, no correction)
    can land on a reflection (determinant -1) — that happens
    whenever a mirror image fits the landmarks better than any rotation
    does, which would indicate bad user input (image flipped).
    """
    mirror_across_z = np.diag([1.0, 1.0, -1.0])
    target = REFERENCE_POINTS @ mirror_across_z.T

    with caplog.at_level(logging.WARNING, logger="core_model.registration"):
        transform = fit_rigid_transform(REFERENCE_POINTS, target)

    det = np.linalg.det(transform.rotation)
    assert det == pytest.approx(1.0, abs=1e-6)
    assert transform.reflection_detected is True
    # a true mirror relationship can't be exactly reproduced by any
    # rotation, so some residual error is expected here (unlike the
    # perfect-fit rotation/translation tests above).
    assert registration_rmse(transform, REFERENCE_POINTS, target) > 1.0
    assert "reflection" in caplog.text


def test_identity_when_source_equals_target():
    transform = fit_rigid_transform(REFERENCE_POINTS, REFERENCE_POINTS)

    assert np.allclose(transform.rotation, np.eye(3), atol=1e-8)
    assert np.allclose(transform.translation, np.zeros(3), atol=1e-8)
    assert transform.rotation_angle_deg() == pytest.approx(0.0, abs=1e-6)


def test_rmse_is_positive_for_imperfect_correspondence():
    target = REFERENCE_POINTS + np.array([1.0, 0.0, 0.0])
    # perturb one point so the correspondence is no longer perfectly rigid
    target[0] += np.array([2.0, 0.0, 0.0])

    transform = fit_rigid_transform(REFERENCE_POINTS, target)

    assert registration_rmse(transform, REFERENCE_POINTS, target) > 0.1


def test_rejects_fewer_than_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        fit_rigid_transform(REFERENCE_POINTS[:2], REFERENCE_POINTS[:2])


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        fit_rigid_transform(REFERENCE_POINTS, REFERENCE_POINTS[:3])


def test_rejects_wrong_dimensionality():
    points_2d = REFERENCE_POINTS[:, :2]
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        fit_rigid_transform(points_2d, points_2d)


def test_apply_is_consistent_with_fit():
    true_rotation = rotation_about_z(-90.0)
    true_translation = np.array([1.0, 2.0, 3.0])
    target = REFERENCE_POINTS @ true_rotation.T + true_translation

    transform = fit_rigid_transform(REFERENCE_POINTS, target)
    aligned = transform.apply(REFERENCE_POINTS)

    assert np.allclose(aligned, target, atol=1e-6)
