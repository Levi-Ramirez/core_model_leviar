from concurrent.futures import Future
import os
from pathlib import Path
import subprocess
import sys

from matplotlib.figure import Figure
import numpy as np
import pytest
import trimesh
from PIL import Image

from core_model import render
from core_model.anatomy import SUPPORTED_STRUCTURES
from core_model.constants.anatomy_constants import SPINE_AND_PELVIS_STRUCTURES
from core_model.render import _render_one, render_previews

_NO_REFERENCE_INDIVIDUAL = "Skeletal reference unavailable."
_NO_REFERENCE_COMBINED = "Skeletal reference unavailable; organs may hide one another."
_WITH_REFERENCE_INDIVIDUAL = "Translucent bone is a spatial reference."
_WITH_REFERENCE_COMBINED = "Translucent bone is a spatial reference; organs may hide one another."
# The camera that design.md specifies, independent of the code under test.
_ELEVATION_DEGREES, _AZIMUTH_DEGREES = 18, 65


def _sphere(center=(0, 0, 0), radius=10, subdivisions=1):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_translation(center)
    return mesh


def _spheres(*centers, radius=40):
    return trimesh.util.concatenate([_sphere(center, radius=radius) for center in centers])


def _scene(**meshes):
    scene = trimesh.Scene()
    for name, mesh in meshes.items():
        scene.add_geometry(mesh, geom_name=name)
    return scene


def _toward_camera():
    elevation, azimuth = np.radians((_ELEVATION_DEGREES, _AZIMUTH_DEGREES))
    return np.array((np.cos(elevation) * np.cos(azimuth),
                     np.cos(elevation) * np.sin(azimuth), np.sin(elevation)))


def _expected_width_over_height(mesh):
    """Project the mesh independently of the renderer to get its outline's shape."""
    elevation, azimuth = np.radians((_ELEVATION_DEGREES, _AZIMUTH_DEGREES))
    right = np.array((-np.sin(azimuth), np.cos(azimuth), 0))
    up = np.array((-np.sin(elevation) * np.cos(azimuth),
                   -np.sin(elevation) * np.sin(azimuth), np.cos(elevation)))
    return np.ptp(mesh.vertices @ right) / np.ptp(mesh.vertices @ up)


# Top-level so a real worker process can import them by name.
def _crash_worker(*_args):
    os._exit(1)


def _missing_renderer(*_args):
    raise ImportError("libGL")


def _failed_preview(*_args):
    raise ValueError("bad preview")


class _InlinePool:
    """Run jobs in this process so tests can observe what each preview drew."""

    def __init__(self, **_options):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


@pytest.fixture
def inline(monkeypatch):
    monkeypatch.setattr(render, "ProcessPoolExecutor", _InlinePool)


def _capture_previews(monkeypatch, draw):
    """Record each preview's layers and page before its figure closes."""
    captured = {}
    drawn = {}
    savefig = Figure.savefig

    def capture_draw(layers):
        image = draw(layers)
        drawn.update(layers=layers, drawn_size=image.shape[0])
        return image

    def capture_page(figure, path, **kwargs):
        axes, *inset = figure.axes
        captured[Path(path).stem] = {
            **drawn,
            "axes_extent": axes.get_window_extent(),
            "legend": [text.get_text() for text in figure.legends[0].get_texts()],
            "legend_extent": figure.legends[0].get_window_extent(),
            "texts": [text.get_text() for text in figure.texts],
            "inset": inset,
        }
        return savefig(figure, path, **kwargs)

    monkeypatch.setattr(render, "_draw_anatomy", capture_draw)
    monkeypatch.setattr(Figure, "savefig", capture_page)
    return captured


@pytest.fixture
def page_views(monkeypatch, inline):
    """Check selection and page layout with a placeholder image, without OpenGL."""
    return _capture_previews(monkeypatch, lambda _layers: np.zeros((4, 4, 3), dtype=np.uint8))


@pytest.fixture
def rendered_views(monkeypatch, inline, opengl):
    """Check what PyVista actually draws."""
    return _capture_previews(monkeypatch, render._draw_anatomy)


def _anatomy_image(path, view):
    """Return the RGB anatomy image, without the orientation guide over its corner."""
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGB"))
    height = pixels.shape[0]
    guide = view["inset"][0].get_window_extent()
    pixels[height - round(guide.y1):height - round(guide.y0), round(guide.x0):round(guide.x1)] = 255
    box = view["axes_extent"]
    return pixels[height - round(box.y1):height - round(box.y0), round(box.x0):round(box.x1)]


def _anatomy_mask(image):
    return image.min(axis=2) < 250


def _render_shape(tmp_path, rendered_views, mesh):
    path = tmp_path / "liver.png"
    _render_one([("liver", mesh)], path, "liver", False)
    return _anatomy_image(path, rendered_views["liver"]), rendered_views["liver"]


def test_worker_process_writes_combined_and_individual_1200px_previews(tmp_path, opengl):
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


def test_worker_process_creates_missing_output_directory_from_string_path(tmp_path, opengl):
    target = tmp_path / "nested" / "previews"

    result = render_previews(_scene(liver=_sphere()), str(target))

    assert result.paths["liver"] == target / "liver.png"
    assert result.paths["liver"].is_file()


def test_renderer_crash_is_reported_and_does_not_end_the_caller(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "_render_one", _crash_worker)

    result = render_previews(_scene(liver=_sphere(), spleen=_sphere((30, 0, 0))), tmp_path)

    assert result.paths == {}
    assert result.failures == dict.fromkeys(("combined", "liver", "spleen"), "renderer_unavailable")


def test_renderer_that_will_not_load_is_reported_as_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "_render_one", _missing_renderer)

    result = render_previews(_scene(liver=_sphere()), tmp_path)

    assert result.failures == dict.fromkeys(("combined", "liver"), "renderer_unavailable")


def test_other_worker_error_reports_preview_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "_render_one", _failed_preview)

    result = render_previews(_scene(liver=_sphere()), tmp_path)

    assert result.failures == dict.fromkeys(("combined", "liver"), "preview_failed")


def test_worker_that_cannot_be_created_keeps_the_run_partial(tmp_path, monkeypatch):
    def no_resources(**_options):
        raise OSError("too many open files")

    monkeypatch.setattr(render, "ProcessPoolExecutor", no_resources)

    result = render_previews(_scene(liver=_sphere(), spleen=_sphere((30, 0, 0))), tmp_path)

    assert result.paths == {}
    assert result.failures == dict.fromkeys(("combined", "liver", "spleen"), "renderer_unavailable")


def test_importing_render_does_not_load_the_renderer():
    code = ("import sys, core_model.render; "
            "assert not {'pyvista', 'vtk', 'vtkmodules'} & set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_render_reports_failed_preview_without_mutating_mesh(tmp_path, page_views):
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


def test_empty_scene_reports_failed_combined_preview(tmp_path, page_views):
    result = render_previews(trimesh.Scene(), tmp_path)

    assert result.paths == {}
    assert result.failures == {"combined": "preview_failed"}


def test_individual_preview_has_skeletal_context_and_preserves_meshes(tmp_path, page_views):
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
    view = page_views["liver"]
    assert [opacity for _, _, opacity in view["layers"]] == [1.0, 0.2, 0.2]
    assert view["legend"] == ["Liver", "Spine / pelvis reference"]
    assert view["texts"][0] == "Liver"
    assert view["texts"][-1] == _WITH_REFERENCE_INDIVIDUAL
    for name, mesh in scene.geometry.items():
        for actual, expected in zip((mesh.vertices, mesh.faces, mesh.visual.vertex_colors), originals[name]):
            np.testing.assert_array_equal(actual, expected)


def test_surfaces_are_shaded_by_their_direction(tmp_path, rendered_views):
    image, _ = _render_shape(tmp_path, rendered_views, _sphere(radius=100, subdivisions=3))

    # Only the sphere's middle, so soft edge pixels cannot satisfy this.
    size = image.shape[0]
    middle = image[size // 3:2 * size // 3, size // 3:2 * size // 3].astype(int)
    assert _anatomy_mask(image[size // 3:2 * size // 3, size // 3:2 * size // 3]).all()
    assert np.ptp(middle.sum(axis=2)) > 30


def test_orientation_guide_uses_the_anatomy_camera(tmp_path, page_views):
    render_previews(_scene(liver=_sphere()), tmp_path)

    guide = page_views["liver"]["inset"]
    assert len(guide) == 1
    assert (guide[0].elev, guide[0].azim) == (_ELEVATION_DEGREES, _AZIMUTH_DEGREES)
    assert [text.get_text() for text in guide[0].texts] == ["R", "A", "S"]


def test_individual_bone_preview_is_opaque_blue_over_other_reference_bones(tmp_path, page_views):
    scene = _scene(vertebrae_L1=_sphere((0, 0, 20)), hip_left=_sphere((-20, 0, -20)))

    result = render_previews(scene, tmp_path, names=("vertebrae_L1",))

    assert not result.failures
    view = page_views["vertebrae_L1"]
    assert len(view["layers"]) == 2  # the selected bone is not drawn again as reference
    selected, reference = view["layers"]
    assert selected[2] == 1.0
    np.testing.assert_array_equal(selected[1], render._SELECTED_BONE_COLOR)
    assert reference[2] == render._CONTEXT_ALPHA
    assert view["texts"][0] == "Vertebrae L1"
    assert view["legend"] == ["Vertebrae L1", "Spine / pelvis reference"]


def test_only_bone_in_scene_has_no_skeletal_reference(tmp_path, page_views):
    render_previews(_scene(sacrum=_sphere()), tmp_path, names=("sacrum",))

    assert page_views["sacrum"]["legend"] == ["Sacrum"]
    assert page_views["sacrum"]["texts"][-1] == _NO_REFERENCE_INDIVIDUAL


def test_combined_preview_draws_unnamed_bones_as_reference_without_own_preview(tmp_path, page_views):
    scene = _scene(liver=_sphere((30, 0, 0)), sacrum=_sphere(), hip_left=_sphere((-30, 0, 0)))

    result = render_previews(scene, tmp_path, names=("liver", "sacrum"))

    assert set(result.paths) == {"combined", "liver", "sacrum"}
    view = page_views["combined"]
    assert view["texts"][0] == "Anatomy overview"
    assert [opacity for _, _, opacity in view["layers"]] == [1.0, 0.2, 0.2]
    assert view["legend"] == ["Liver", "Spine / pelvis reference"]
    assert view["texts"][-1] == _WITH_REFERENCE_COMBINED


def test_combined_preview_warns_organs_can_hide_without_bones(tmp_path, page_views):
    render_previews(_scene(liver=_sphere(), spleen=_sphere((5, 0, 0))), tmp_path)

    assert page_views["combined"]["texts"][-1] == _NO_REFERENCE_COMBINED
    assert page_views["liver"]["texts"][-1] == _NO_REFERENCE_INDIVIDUAL


def test_names_are_deduplicated_and_absent_names_are_skipped(tmp_path, page_views):
    scene = _scene(liver=_sphere(), spleen=_sphere((30, 0, 0)))

    result = render_previews(scene, tmp_path, names=("spleen", "liver", "spleen", "pancreas"))

    assert list(result.paths) == ["combined", "spleen", "liver"]
    assert result.failures == {}
    assert page_views["combined"]["legend"] == ["Spleen", "Liver"]
    assert not (tmp_path / "pancreas.png").exists()


def test_unknown_structure_name_fails_only_its_previews(tmp_path, page_views):
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
def test_nonfinite_bone_does_not_fail_other_previews(tmp_path, page_views, names, expected_failures):
    bad_bone = _sphere()
    bad_bone.vertices[0, 0] = np.nan
    scene = _scene(liver=_sphere((30, 0, 0)), spleen=_sphere((60, 0, 0)),
                   hip_left=_sphere((-30, 0, 0)), sacrum=bad_bone)

    result = render_previews(scene, tmp_path, names=names)

    assert set(result.failures) == expected_failures
    assert {"liver", "spleen"} <= set(result.paths)
    assert len(page_views["liver"]["layers"]) == 2  # organ plus the one finite bone


def test_patient_right_is_image_left_and_superior_is_up(tmp_path, rendered_views):
    scene = _scene(liver=_sphere((100, 0, 100), radius=30, subdivisions=2),
                   spleen=_sphere((-100, 0, -100), radius=30, subdivisions=2))

    result = render_previews(scene, tmp_path, names=("liver", "spleen"))

    image = _anatomy_image(result.paths["combined"], rendered_views["combined"]).astype(int)
    red, blue = image[..., 0], image[..., 2]
    liver_rows, liver_columns = np.nonzero(red - blue > 30)
    spleen_rows, spleen_columns = np.nonzero(blue - red > 10)
    assert liver_columns.mean() < spleen_columns.mean()
    assert liver_rows.mean() < spleen_rows.mean()


def test_anterior_surface_is_nearer_the_camera(tmp_path, rendered_views):
    # Overlapping on screen, so whichever is nearer shows at the midpoint.
    scene = _scene(liver=_sphere((0, 120, 0), radius=100, subdivisions=2),
                   spleen=_sphere((0, -120, 0), radius=100, subdivisions=2))

    result = render_previews(scene, tmp_path, names=("liver", "spleen"))

    image = _anatomy_image(result.paths["combined"], rendered_views["combined"]).astype(int)
    red, _, blue = image[image.shape[0] // 2, image.shape[1] // 2]
    assert red > blue  # the anterior liver, not the posterior spleen


@pytest.mark.parametrize("names", [("liver", "spleen"), ("spleen", "liver")],
                         ids=["near_first", "far_first"])
def test_nearer_surface_hides_farther_surface_in_any_draw_order(tmp_path, rendered_views, names):
    near = _sphere(_toward_camera() * 100, radius=50, subdivisions=2)
    far = _sphere(-_toward_camera() * 100, radius=50, subdivisions=2)
    scene = _scene(liver=near, spleen=far)

    result = render_previews(scene, tmp_path, names=names)

    image = _anatomy_image(result.paths["combined"], rendered_views["combined"]).astype(int)
    center = image[image.shape[0] // 2, image.shape[1] // 2]
    assert center[0] > center[2]  # the liver's red, not the spleen's purple


def test_organ_behind_translucent_bone_stays_visible(tmp_path, rendered_views):
    organ = _sphere(-_toward_camera() * 100, radius=50, subdivisions=2)
    bone = _sphere(_toward_camera() * 100, radius=80, subdivisions=2)

    result = render_previews(_scene(liver=organ, hip_left=bone), tmp_path, names=("liver",))

    image = _anatomy_image(result.paths["liver"], rendered_views["liver"]).astype(int)
    red, _, blue = image[image.shape[0] // 2, image.shape[1] // 2]
    assert red - blue > 25  # the organ tints the view through the bone


@pytest.mark.parametrize(
    "mesh",
    [
        _sphere((500, -300, 200), radius=100, subdivisions=3),
        _spheres((-200, 150, -100), (200, -150, 300)),
        _spheres((0, 0, 0), (300, 0, 0), (300, 0, 400), radius=60),
        # Outline centers differ from the bounds' center across and up the screen.
        _spheres((-200, 0, 0), (200, 0, 0), (0, 300, 0)),
        _spheres((0, -200, 0), (0, 200, 0), (0, 200, 300)),
        trimesh.creation.box(extents=(100, 100, 100)),
        trimesh.creation.box(extents=(100, 100, 800)),
        trimesh.creation.box(extents=(800, 100, 100)),
        trimesh.creation.box(extents=(100, 800, 100)),
        trimesh.creation.box(extents=(800, 100, 800)),
    ],
    ids=["offset_sphere", "diagonal", "l_shape", "off_center_across", "off_center_up",
         "cube", "tall", "wide", "deep", "slab"],
)
def test_outline_fills_the_view_centered_at_true_proportions_without_clipping(tmp_path, rendered_views, mesh):
    image, _ = _render_shape(tmp_path, rendered_views, mesh)

    height, width = image.shape[:2]
    mask = _anatomy_mask(image)
    rows, columns = np.nonzero(mask)
    # The rendered outline keeps the shape the camera direction should produce.
    assert (np.ptp(columns) + 1) / (np.ptp(rows) + 1) == pytest.approx(
        _expected_width_over_height(mesh), rel=0.03)
    # Sized from the visible outline, not from the render's empty surroundings.
    assert max(np.ptp(rows), np.ptp(columns)) > 0.94 * max(height, width)
    assert (rows.min() + rows.max()) / 2 == pytest.approx(height / 2, abs=3)
    assert (columns.min() + columns.max()) / 2 == pytest.approx(width / 2, abs=3)
    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        assert not edge.any()


def test_anatomy_is_drawn_larger_than_shown_so_scaling_smooths_edges(tmp_path, rendered_views):
    _render_shape(tmp_path, rendered_views, _sphere(radius=100, subdivisions=3))

    view = rendered_views["liver"]
    assert view["drawn_size"] > 1.2 * view["axes_extent"].height


def test_crop_centers_off_center_anatomy_in_a_padded_square():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    image[10:30, 150:190] = 0  # 20 rows by 40 columns, near a corner

    cropped = render._crop_to_anatomy(image)

    assert cropped.shape == (42, 42, 3)
    rows, columns = np.nonzero(cropped.min(axis=2) < 255)
    assert (rows.min(), rows.max()) == (11, 30)
    assert (columns.min(), columns.max()) == (1, 40)


def test_crop_rejects_a_blank_render():
    with pytest.raises(ValueError, match="empty_render"):
        render._crop_to_anatomy(np.full((10, 10, 3), 255, dtype=np.uint8))


def test_sphere_stays_round_at_equal_physical_scale(tmp_path, rendered_views):
    image, _ = _render_shape(tmp_path, rendered_views, _sphere(radius=100, subdivisions=3))

    rows, columns = np.nonzero(_anatomy_mask(image))
    assert np.ptp(rows) == pytest.approx(np.ptp(columns), rel=0.02)


def test_thin_plane_renders(tmp_path, rendered_views):
    plane = trimesh.Trimesh(vertices=[(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)],
                            faces=[(0, 1, 2), (0, 2, 3)])

    image, _ = _render_shape(tmp_path, rendered_views, plane)

    assert _anatomy_mask(image).any()


def test_mesh_that_draws_nothing_reports_a_failed_preview(tmp_path, rendered_views):
    point = trimesh.Trimesh(vertices=[(5, 5, 5)] * 3, faces=[(0, 1, 2)], process=False)

    result = render_previews(_scene(liver=point), tmp_path, names=("liver",))

    assert result.failures == {"combined": "preview_failed", "liver": "preview_failed"}


def test_short_legend_keeps_the_default_anatomy_area(tmp_path, page_views):
    render_previews(_scene(liver=_sphere()), tmp_path)

    view = page_views["liver"]
    assert view["axes_extent"].y0 == pytest.approx(0.10 * 1200)
    assert view["legend_extent"].y1 < view["axes_extent"].y0


def test_long_legend_does_not_cover_the_anatomy(tmp_path, page_views):
    organs = [name for name in SUPPORTED_STRUCTURES if name not in SPINE_AND_PELVIS_STRUCTURES]
    meshes = [(name, _sphere((index * 5, 0, 0))) for index, name in enumerate(organs)]

    # One overview exercises the layout without 90 individual previews.
    _render_one(meshes, tmp_path / "combined.png", "combined", True)

    view = page_views["combined"]
    assert len(view["legend"]) == len(organs)
    assert view["legend_extent"].y1 < view["axes_extent"].y0
    assert view["axes_extent"].height > 0
