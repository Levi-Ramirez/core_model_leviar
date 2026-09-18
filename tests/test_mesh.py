import nibabel as nib
import numpy as np
import pytest

from core_model.mesh import MeshError, extract_mesh


def write_mask(path, data, affine=None, units="mm"):
    image = nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units(units)
    nib.save(image, path)


def test_extract_mesh_maps_voxels_to_world_and_preserves_boundary_components(tmp_path):
    data = np.zeros((8, 8, 8), dtype=np.uint8)
    data[2:4, 3:6, 1:3] = 1
    data[6, 6, 6] = 1
    affine = np.array([[2, 0, 0, 10], [0, 3, 0, 20], [0, 0, 4, 30], [0, 0, 0, 1]], float)
    path = tmp_path / "mask.nii.gz"
    write_mask(path, data, affine)

    result = extract_mesh(path, expected_shape=data.shape, expected_affine=affine)

    assert result.clipped is False
    assert len(result.mesh.split(only_watertight=False)) == 2
    np.testing.assert_allclose(result.mesh.bounds[0], [13, 27.5, 32], atol=1e-6)
    np.testing.assert_allclose(result.mesh.bounds[1], [23, 39.5, 56], atol=1e-6)


def test_extract_mesh_closes_boundary_and_reports_clipping(tmp_path):
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[0:2, 1:3, 1:3] = 1
    path = tmp_path / "mask.nii.gz"
    write_mask(path, data)
    result = extract_mesh(path)
    assert result.clipped is True
    assert result.warnings == ("clipped_to_scan_boundary",)
    assert result.mesh.is_watertight


def test_extract_mesh_reverses_winding_for_reflected_affine(tmp_path):
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[1:4, 1:4, 1:4] = 1
    affine = np.diag([-2, 2, 2, 1])
    path = tmp_path / "mask.nii.gz"
    write_mask(path, data, affine)
    result = extract_mesh(path)
    assert np.linalg.det(affine[:3, :3]) < 0
    assert result.mesh.is_watertight


@pytest.mark.parametrize("data, reason", [
    (np.zeros((3, 3, 3), dtype=np.uint8), "empty_mask"),
    (np.full((3, 3, 3), 2, dtype=np.uint8), "nonbinary_mask"),
])
def test_extract_mesh_rejects_invalid_masks(tmp_path, data, reason):
    path = tmp_path / "mask.nii.gz"
    write_mask(path, data)
    with pytest.raises(MeshError, match=reason):
        extract_mesh(path)


def test_extract_mesh_rejects_grid_mismatch(tmp_path):
    path = tmp_path / "mask.nii.gz"
    write_mask(path, np.ones((3, 3, 3), dtype=np.uint8))
    with pytest.raises(MeshError, match="mask_grid_mismatch"):
        extract_mesh(path, expected_shape=(4, 4, 4))
