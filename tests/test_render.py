from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pytest
import trimesh
from PIL import Image

from core_model.anatomy import SUPPORTED_STRUCTURES
from core_model.constants.anatomy_constants import SPINE_AND_PELVIS_STRUCTURES
from core_model.render import _render_one, render_previews

_NO_REFERENCE_INDIVIDUAL = "Skeletal reference unavailable in this view."
_NO_REFERENCE_COMBINED = "Skeletal reference unavailable; surface overlap is approximate."
_WITH_REFERENCE = "Translucent bone is a spatial reference; surface overlap is approximate."


def _sphere(center=(0, 0, 0), radius=10, subdivisions=1):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_translation(center)
    return mesh


def _scene(**meshes):
    scene = trimesh.Scene()
    for name, mesh in meshes.items():
        scene.add_geometry(mesh, geom_name=name)
    return scene


def _pixels_inside_axes(path, view):
    """Return grayscale pixels of the anatomy axes, without the orientation guide."""
    with Image.open(path) as image:
        pixels = np.array(image.convert("L"))
    height = pixels.shape[0]
    # The orientation guide legitimately overlaps a corner of the anatomy axes.
    guide = view["inset"][0].get_window_extent()
    pixels[height - round(guide.y1):height - round(guide.y0), round(guide.x0):round(guide.x1)] = 255
    box = view["axes_extent"]
    return pixels[height - round(box.y1):height - round(box.y0), round(box.x0):round(box.x1)]


@pytest.fixture
def views(monkeypatch):
    """Record what each figure drew, since figures close right after saving."""
    captured = {}
    savefig = Figure.savefig

    def capture(figure, path, **kwargs):
        axes, *inset = figure.axes
        captured[Path(path).stem] = {
            "colors": [np.asarray(c.get_facecolor()) for c in axes.collections],
            "elevation": axes.elev,
            "azimuth": axes.azim,
            "limits": np.array([axes.get_xlim(), axes.get_ylim(), axes.get_zlim()]),
            "aspect": axes.get_box_aspect(),
            "axes_extent": axes.get_window_extent(),
            "legend": [text.get_text() for text in figure.legends[0].get_texts()],
            "legend_extent": figure.legends[0].get_window_extent(),
            "texts": [text.get_text() for text in figure.texts],
            "inset": inset,
        }
        return savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return captured


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


def test_render_creates_missing_output_directory_from_string_path(tmp_path):
    target = tmp_path / "nested" / "previews"

    result = render_previews(_scene(liver=_sphere()), str(target))

    assert result.paths["liver"] == target / "liver.png"
    assert result.paths["liver"].is_file()


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


def test_empty_scene_reports_failed_combined_preview(tmp_path):
    result = render_previews(trimesh.Scene(), tmp_path)

    assert result.paths == {}
    assert result.failures == {"combined": "preview_failed"}


def test_individual_preview_has_shaded_skeletal_context_and_preserves_meshes(tmp_path, views):
    scene = trimesh.Scene()
    for name, center in (("liver", (30, 40, 15)), ("vertebrae_L1", (0, 0, 20)),
                         ("hip_left", (-20, 0, -20))):
        mesh = _sphere(center)
        mesh.visual.vertex_colors = [90, 120, 150, 255]
        scene.add_geometry(mesh, geom_name=name)
    originals = {name: (mesh.vertices.copy(), mesh.faces.copy(), mesh.visual.vertex_colors.copy())
                 for name, mesh in scene.geometry.items()}

    result = render_previews(scene, tmp_path, names=("liver",))

    assert not result.failures
    view = views["liver"]
    assert len(view["colors"]) == 3  # organ plus both skeletal landmarks
    assert np.ptp(view["colors"][0][:, :3], axis=0).max() > 0.1
    assert np.all(view["colors"][0][:, 3] == 1)
    assert all(np.all((c[:, 3] > 0) & (c[:, 3] < 1)) for c in view["colors"][1:])
    assert 0 < view["elevation"] < 45
    assert 0 < view["azimuth"] < 90  # camera is on the anterior (+Y) side
    assert view["legend"] == ["Liver", "Spine / pelvis reference"]
    assert view["texts"][0] == "Liver"
    assert view["texts"][-1] == _WITH_REFERENCE
    for name, mesh in scene.geometry.items():
        for actual, expected in zip((mesh.vertices, mesh.faces, mesh.visual.vertex_colors), originals[name]):
            np.testing.assert_array_equal(actual, expected)


def test_orientation_guide_uses_the_anatomy_camera(tmp_path, views):
    render_previews(_scene(liver=_sphere()), tmp_path)

    view = views["liver"]
    assert len(view["inset"]) == 1
    guide = view["inset"][0]
    assert (guide.elev, guide.azim) == (view["elevation"], view["azimuth"])
    assert [text.get_text() for text in guide.texts] == ["R", "A", "S"]


def test_individual_bone_preview_is_opaque_blue_over_other_reference_bones(tmp_path, views):
    scene = _scene(vertebrae_L1=_sphere((0, 0, 20)), hip_left=_sphere((-20, 0, -20)))

    result = render_previews(scene, tmp_path, names=("vertebrae_L1",))

    assert not result.failures
    view = views["vertebrae_L1"]
    assert len(view["colors"]) == 2  # the selected bone is not drawn again as reference
    selected, reference = view["colors"]
    assert np.all(selected[:, 3] == 1)
    assert np.all(selected[:, 2] > selected[:, 0])  # blue dominates after shading
    assert np.all(reference[:, 3] < 1)
    assert view["texts"][0] == "Vertebrae L1"  # case in the level name is preserved
    assert view["legend"] == ["Vertebrae L1", "Spine / pelvis reference"]


def test_only_bone_in_scene_has_no_skeletal_reference(tmp_path, views):
    render_previews(_scene(sacrum=_sphere()), tmp_path, names=("sacrum",))

    assert views["sacrum"]["legend"] == ["Sacrum"]
    assert views["sacrum"]["texts"][-1] == _NO_REFERENCE_INDIVIDUAL


def test_combined_preview_draws_unnamed_bones_as_reference_without_own_preview(tmp_path, views):
    scene = _scene(liver=_sphere((30, 0, 0)), sacrum=_sphere(), hip_left=_sphere((-30, 0, 0)))

    result = render_previews(scene, tmp_path, names=("liver", "sacrum"))

    assert set(result.paths) == {"combined", "liver", "sacrum"}
    view = views["combined"]
    assert view["texts"][0] == "Anatomy overview"
    assert len(view["colors"]) == 3  # hip_left is drawn even though it is not named
    assert np.all(view["colors"][0][:, 3] == 1)
    assert all(np.all(colors[:, 3] < 1) for colors in view["colors"][1:])
    assert view["legend"] == ["Liver", "Spine / pelvis reference"]
    assert view["texts"][-1] == _WITH_REFERENCE


def test_combined_preview_keeps_overlap_caveat_without_bones(tmp_path, views):
    render_previews(_scene(liver=_sphere(), spleen=_sphere((5, 0, 0))), tmp_path)

    assert views["combined"]["texts"][-1] == _NO_REFERENCE_COMBINED
    assert views["liver"]["texts"][-1] == _NO_REFERENCE_INDIVIDUAL


def test_names_are_deduplicated_and_absent_names_are_skipped(tmp_path, views):
    scene = _scene(liver=_sphere(), spleen=_sphere((30, 0, 0)))

    result = render_previews(scene, tmp_path, names=("spleen", "liver", "spleen", "pancreas"))

    assert list(result.paths) == ["combined", "spleen", "liver"]
    assert result.failures == {}
    assert views["combined"]["legend"] == ["Spleen", "Liver"]
    assert not (tmp_path / "pancreas.png").exists()


def test_unknown_structure_name_fails_only_its_previews(tmp_path):
    scene = _scene(liver=_sphere(), not_a_structure=_sphere((30, 0, 0)))

    result = render_previews(scene, tmp_path)

    assert result.failures == {"combined": "preview_failed", "not_a_structure": "preview_failed"}
    assert set(result.paths) == {"liver"}


@pytest.mark.parametrize(
    "names, expected_failures",
    [
        (("liver", "spleen"), set()),
        (("liver", "spleen", "sacrum"), {"combined", "sacrum"}),
    ],
    ids=["unnamed_bone", "named_bone"],
)
def test_nonfinite_bone_does_not_fail_other_previews(tmp_path, views, names, expected_failures):
    bad_bone = _sphere()
    bad_bone.vertices[0, 0] = np.nan
    scene = _scene(liver=_sphere((30, 0, 0)), spleen=_sphere((60, 0, 0)),
                   hip_left=_sphere((-30, 0, 0)), sacrum=bad_bone)

    result = render_previews(scene, tmp_path, names=names)

    assert set(result.failures) == expected_failures
    assert {"liver", "spleen"} <= set(result.paths)
    assert len(views["liver"]["colors"]) == 2  # organ plus the one finite bone


def test_equal_physical_scale_with_non_cubic_bounds(tmp_path, views):
    scene = _scene(liver=trimesh.creation.box(extents=(10, 40, 200)))

    render_previews(scene, tmp_path, names=("liver",))

    view = views["liver"]
    spans = np.diff(view["limits"], axis=1).ravel()
    units_per_mm = view["aspect"] / spans
    np.testing.assert_allclose(units_per_mm, units_per_mm[0])
    # Bounds are preserved and padded rather than stretched to a cube.
    np.testing.assert_allclose(spans, np.array((16, 40, 200)) * 1.04)


def test_thin_structure_keeps_a_minimum_view_depth(tmp_path, views):
    plane = trimesh.Trimesh(vertices=[(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)],
                            faces=[(0, 1, 2), (0, 2, 3)])

    result = render_previews(_scene(liver=plane), tmp_path, names=("liver",))

    assert not result.failures
    spans = np.diff(views["liver"]["limits"], axis=1).ravel()
    np.testing.assert_allclose(spans, np.array((100, 100, 8)) * 1.04)


def test_zero_extent_mesh_still_renders(tmp_path):
    point = trimesh.Trimesh(vertices=[(5, 5, 5)] * 3, faces=[(0, 1, 2)], process=False)

    result = render_previews(_scene(liver=point), tmp_path, names=("liver",))

    assert result.failures == {}
    assert result.paths["liver"].is_file()


@pytest.mark.parametrize(
    "extents",
    [(100, 100, 100), (100, 100, 250), (100, 100, 800), (800, 100, 100), (100, 800, 100),
     (800, 800, 800), (800, 100, 800)],
    ids=["cube", "tall", "very_tall", "wide", "deep", "large_cube", "slab"],
)
def test_elongated_anatomy_is_not_clipped_by_the_axes(tmp_path, views, extents):
    result = render_previews(_scene(liver=trimesh.creation.box(extents=extents)), tmp_path,
                             names=("liver",))

    inside = _pixels_inside_axes(result.paths["liver"], views["liver"])
    # The anatomy is drawn, and every edge of the anatomy axes stays background.
    assert (inside < 200).any()
    for edge in (inside[0], inside[-1], inside[:, 0], inside[:, -1]):
        assert (edge > 250).all()


@pytest.mark.parametrize(
    "mesh",
    [
        _sphere((500, -300, 200), radius=100, subdivisions=3),
        trimesh.util.concatenate([_sphere((-200, 150, -100), radius=40),
                                  _sphere((200, -150, 300), radius=40)]),
        trimesh.util.concatenate([_sphere((0, 0, 0), radius=60),
                                  _sphere((300, 0, 0), radius=60),
                                  _sphere((300, 0, 400), radius=60)]),
        # Outline centers differ from the bounds' center across and up the screen.
        trimesh.util.concatenate([_sphere((-200, 0, 0), radius=40),
                                  _sphere((200, 0, 0), radius=40),
                                  _sphere((0, 300, 0), radius=40)]),
        trimesh.util.concatenate([_sphere((0, -200, 0), radius=40),
                                  _sphere((0, 200, 0), radius=40),
                                  _sphere((0, 200, 300), radius=40)]),
    ],
    ids=["offset_sphere", "diagonal", "l_shape", "off_center_across", "off_center_up"],
)
def test_visible_outline_fills_and_is_centered_in_the_axes(tmp_path, views, mesh):
    result = render_previews(_scene(liver=mesh), tmp_path, names=("liver",))

    inside = _pixels_inside_axes(result.paths["liver"], views["liver"])
    rows, columns = np.nonzero(inside < 250)
    side = min(inside.shape)
    # Sized from the mesh outline, not from the mostly empty 3D bounding box.
    assert max(np.ptp(rows), np.ptp(columns)) > 0.9 * side
    # Matplotlib's view spans -0.95 to +0.9, so screen center sits at 51.35%.
    assert (rows.min() + rows.max()) / 2 == pytest.approx(0.4865 * inside.shape[0], abs=0.02 * side)
    assert (columns.min() + columns.max()) / 2 == pytest.approx(0.5135 * inside.shape[1], abs=0.02 * side)


def test_sphere_stays_round_at_equal_physical_scale(tmp_path, views):
    mesh = _sphere(radius=100, subdivisions=3)
    result = render_previews(_scene(liver=mesh), tmp_path, names=("liver",))

    rows, columns = np.nonzero(_pixels_inside_axes(result.paths["liver"], views["liver"]) < 250)
    assert np.ptp(rows) == pytest.approx(np.ptp(columns), rel=0.02)


def test_short_legend_keeps_the_default_anatomy_area(tmp_path, views):
    render_previews(_scene(liver=_sphere()), tmp_path)

    view = views["liver"]
    assert view["axes_extent"].y0 == pytest.approx(0.10 * 1200)
    assert view["legend_extent"].y1 < view["axes_extent"].y0


def test_long_legend_does_not_cover_the_anatomy(tmp_path, views):
    organs = [name for name in SUPPORTED_STRUCTURES if name not in SPINE_AND_PELVIS_STRUCTURES]
    meshes = [(name, _sphere((index * 5, 0, 0))) for index, name in enumerate(organs)]

    # One overview exercises the layout without 90 individual previews.
    _render_one(meshes, tmp_path / "combined.png", "combined", True)

    view = views["combined"]
    assert len(view["legend"]) == len(organs)
    assert view["legend_extent"].y1 < view["axes_extent"].y0
    assert view["axes_extent"].height > 0
