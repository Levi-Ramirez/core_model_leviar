import numpy as np
import trimesh
from PIL import Image

from core_model.render import render_previews


def test_render_writes_combined_and_individual_1200px_previews(tmp_path):
    scene = trimesh.Scene()
    liver = trimesh.creation.box(extents=(2, 4, 6))
    liver.apply_translation((0, 0, 3))
    scene.add_geometry(liver, node_name="liver", geom_name="liver")
    result = render_previews(scene, tmp_path)

    assert result.failures == {}
    assert set(result.paths) == {"combined", "liver"}
    with Image.open(result.paths["combined"]) as image:
        assert image.size == (1200, 1200)
        assert np.asarray(image).std() > 0


def test_render_reports_failed_preview_without_mutating_mesh(tmp_path):
    scene = trimesh.Scene()
    mesh = trimesh.creation.icosphere()
    bad = trimesh.creation.box()
    bad.vertices[0, 0] = np.nan
    vertices = mesh.vertices.copy()
    scene.add_geometry(mesh, node_name="liver", geom_name="liver")
    scene.add_geometry(bad, node_name="spleen", geom_name="spleen")
    result = render_previews(scene, tmp_path)

    assert result.failures == {"combined": "preview_failed", "spleen": "preview_failed"}
    assert "liver" in result.paths
    np.testing.assert_array_equal(mesh.vertices, vertices)
