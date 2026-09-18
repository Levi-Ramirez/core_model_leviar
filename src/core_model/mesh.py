"""Extract full-fidelity physical meshes from validated segmentation masks."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
import trimesh

from core_model.constants.ingest_constants import FORM_TOLERANCE_MM, UNIT_SCALE_TO_MM
from core_model.contracts import GenerationError


class MeshError(GenerationError):
    """Safe error raised when a mask cannot produce a physical mesh.

    Input:
        reason: Stable, non-identifying mesh error code.
    Output:
        An exception carrying the safe reason for the mesh stage.
    """

    def __init__(self, reason: str):
        super().__init__("mesh", reason)


@dataclass(frozen=True)
class MeshResult:
    """A mesh and the coverage warning associated with its source mask.

    Input:
        mesh: Full-fidelity RAS+ millimeter ``trimesh.Trimesh`` geometry.
        clipped: Whether foreground touched the source image boundary.
        warnings: Safe messages describing coverage limitations.
    Output:
        An immutable extraction result for model assembly.
    """

    mesh: trimesh.Trimesh
    clipped: bool
    warnings: tuple[str, ...] = ()


def _load_mask(path: Path):
    """Load and validate one binary mask and return its mm data and affine.

    Input: A NIfTI mask path.
    Output: ``(binary_data, affine_mm)`` or a ``MeshError``.
    """
    try:
        image = nib.load(path)
        if len(image.shape) != 3 or any(size < 2 for size in image.shape):
            raise MeshError("invalid_mask_dimensions")
        unit, _ = image.header.get_xyzt_units()
        if unit not in UNIT_SCALE_TO_MM:
            raise MeshError("unknown_spatial_units")
        data = np.asanyarray(image.dataobj)
        affine = np.asarray(image.affine, dtype=float).copy()
    except MeshError:
        raise
    except Exception:
        raise MeshError("invalid_or_unreadable_mask") from None
    if data.ndim != 3 or not np.issubdtype(data.dtype, np.number):
        raise MeshError("non_scalar_mask")
    if not np.isfinite(data).all():
        raise MeshError("nonfinite_mask")
    if not np.isfinite(affine).all() or affine.shape != (4, 4):
        raise MeshError("invalid_affine")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12:
        raise MeshError("singular_affine")
    if not np.allclose(data, np.round(data)) or np.any((data != 0) & (data != 1)):
        raise MeshError("nonbinary_mask")
    scale = UNIT_SCALE_TO_MM[unit]
    affine[:3, :] *= scale
    return data.astype(bool), affine


def _validate_grid(data: np.ndarray, affine: np.ndarray, expected_shape, expected_affine) -> None:
    """Reject masks that do not share the normalized source grid.

    Input:
        data: Three-dimensional mask array.
        affine: Mask affine normalized to millimeters.
        expected_shape: Optional source shape.
        expected_affine: Optional source affine in millimeters.
    Output: None when the grids agree; otherwise raises ``MeshError``.
    """
    if expected_shape is not None and tuple(data.shape) != tuple(expected_shape):
        raise MeshError("mask_grid_mismatch")
    if expected_affine is not None:
        expected = np.asarray(expected_affine, dtype=float)
        if expected.shape != (4, 4) or not np.isfinite(expected).all():
            raise MeshError("invalid_expected_affine")
        corners = np.array(list(product(*((0, n - 1) for n in data.shape))))
        actual_world = nib.affines.apply_affine(affine, corners)
        expected_world = nib.affines.apply_affine(expected, corners)
        if np.any(np.linalg.norm(actual_world - expected_world, axis=1) > FORM_TOLERANCE_MM):
            raise MeshError("mask_grid_mismatch")


def extract_mesh(mask_path: Path, *, expected_shape=None, expected_affine=None) -> MeshResult:
    """Extract a world-coordinate mesh from a binary NIfTI mask.

    Input:
        mask_path: NIfTI mask with a finite, invertible physical affine.
        expected_shape: Optional normalized source-grid shape.
        expected_affine: Optional normalized source affine in millimeters.
    Output:
        A ``MeshResult`` containing unmodified full-fidelity geometry.
    """
    data, affine = _load_mask(Path(mask_path))
    _validate_grid(data, affine, expected_shape, expected_affine)
    if not data.any():
        raise MeshError("empty_mask")
    clipped = bool(
        np.any(data[0]) or np.any(data[-1]) or np.any(data[:, 0]) or
        np.any(data[:, -1]) or np.any(data[:, :, 0]) or np.any(data[:, :, -1])
    )
    # Add a one-voxel background border so marching cubes can close a surface
    # even when foreground touches the original image boundary.
    padded = np.pad(data.astype(np.float32), 1, mode="constant")
    try:
        # The mask contains only 0 and 1, so 0.5 is the boundary between them.
        # Unit spacing keeps these vertices in voxel coordinates for now.
        vertices, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(1, 1, 1))
    except Exception:
        raise MeshError("degenerate_mask_geometry") from None
    # Padding shifted the voxel origin by one index in every direction.
    vertices -= 1.0
    # Convert voxel coordinates to physical RAS+ millimeter coordinates.
    vertices = nib.affines.apply_affine(affine, vertices)
    if not np.isfinite(vertices).all():
        raise MeshError("nonfinite_mesh")
    # A negative determinant means the affine reflects the coordinate system;
    # reverse each triangle so the surface keeps its outward orientation.
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]
    # Keep the extracted geometry unchanged apart from invalid triangle cleanup.
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    # Marching cubes normally produces valid faces, but remove any degenerate
    # triangles defensively before dropping vertices no face references.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise MeshError("degenerate_mesh")
    # Recompute normals after transforming and potentially reversing faces.
    mesh.fix_normals()
    warnings = ("clipped_to_scan_boundary",) if clipped else ()
    return MeshResult(mesh=mesh, clipped=clipped, warnings=warnings)


mask_to_mesh = extract_mesh
