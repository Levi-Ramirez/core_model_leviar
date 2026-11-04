"""Render named model meshes as fixed-view PNG previews.

PyVista draws the anatomy in a worker process and Matplotlib lays out the page
around it. A missing display or OpenGL driver can crash the process that opens
the render window instead of raising, so that process must not be the caller's.
"""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import multiprocessing
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import trimesh

from core_model.anatomy import structure_color, structure_label
from core_model.constants.anatomy_constants import SPINE_AND_PELVIS_STRUCTURES


_CONTEXT = frozenset(SPINE_AND_PELVIS_STRUCTURES)
# RAS+: +Y is anterior, +X is right, +Z is superior.
_ELEVATION, _AZIMUTH = 18, 65
_CONTEXT_COLOR = np.array((0.76, 0.75, 0.71))
_CONTEXT_ALPHA = 0.20
# A mesh stacks front and back faces, so its reference gray looks denser than a
# single legend swatch drawn at the same alpha.
_CONTEXT_LEGEND_ALPHA = 0.4
_SELECTED_BONE_COLOR = np.array((135, 206, 250)) / 255.0  # Light sky blue.

# Drawn larger than the page shows, since anatomy fills only part of the render
# and scaling down smooths its edges.
_RENDER_SIZE_PX = 2400
# Leave 2% of the cropped span as margin on each side.
_FRAME_PADDING_FACTOR = 1.04
# Default vertical span of the anatomy image, as figure fractions.
_ANATOMY_BOTTOM, _ANATOMY_TOP = 0.10, 0.91
_LEGEND_GAP = 0.01
_LEGEND_FONT_SIZE = 12


@dataclass(frozen=True)
class RenderResult:
    """Written preview paths and safe failures for individual previews.

    Input:
        paths: Successfully rendered preview paths by preview name.
        failures: Safe failure reasons by preview name.
    Output:
        An immutable rendering-stage result; valid previews remain available.
    """

    paths: dict[str, Path]
    failures: dict[str, str]


def _crop_to_anatomy(image: np.ndarray) -> np.ndarray:
    """Crop a render to a centered square around its non-white pixels.

    Input: An RGB image on a pure white background.
    Output: A square RGB image with a small margin; raises ``ValueError`` if blank.
    """
    rows, columns = np.nonzero((image < 255).any(axis=2)) # get indices of non-white pixels
    if rows.size == 0:
        raise ValueError("empty_render")
    anatomy = image[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    height, width = anatomy.shape[:2]
    side = round(max(height, width) * _FRAME_PADDING_FACTOR)
    square = np.full((side, side, 3), 255, dtype=np.uint8) # create white square
    top, left = (side - height) // 2, (side - width) // 2
    square[top:top + height, left:left + width] = anatomy # copy anatomy into center of square
    return square


def _draw_anatomy(layers: list[tuple[trimesh.Trimesh, np.ndarray, float]]) -> np.ndarray:
    """Draw meshes from the fixed orthographic camera, cropped to the anatomy.

    Input: ``(mesh, RGB color, opacity)`` layers in RAS+ millimeters.
    Output: A square RGB image on a white background.
    """
    # Imported here so a broken OpenGL install only affects the worker process.
    import pyvista as pv

    pv.vtk_verbosity("off")
    plotter = pv.Plotter(off_screen=True, window_size=(_RENDER_SIZE_PX, _RENDER_SIZE_PX))
    try:
        plotter.set_background("white")
        for mesh, color, opacity in layers:
            surface = pv.PolyData.from_regular_faces(mesh.vertices, mesh.faces)
            plotter.add_mesh(surface, color=color, opacity=opacity, smooth_shading=True,
                             ambient=0.25, diffuse=0.75, specular=0.0)
        elevation, azimuth = np.radians((_ELEVATION, _AZIMUTH)) # convert degree views to rads
        plotter.enable_parallel_projection()
        plotter.view_vector((np.cos(elevation) * np.cos(azimuth),
                             np.cos(elevation) * np.sin(azimuth), np.sin(elevation)),
                            viewup=(0, 0, 1)) # view angle relative to anatomy mesh (viewup defines what dimension is "up")
        return _crop_to_anatomy(plotter.screenshot(return_img=True))
    finally:
        plotter.close()


def _orientation(figure) -> None:
    """Draw world-coordinate directions using the same camera as the anatomy.

    Input: The figure that receives the inset orientation axes.
    Output: None; the figure is updated in place.
    """
    axes = figure.add_axes((0.81, 0.79, 0.15, 0.13), projection="3d")
    axes.view_init(elev=_ELEVATION, azim=_AZIMUTH)
    axes.set_proj_type("ortho")
    for direction, label, color in zip(np.eye(3), ("R", "A", "S"),
                                       ("#b84d49", "#398477", "#4777b0")):
        axes.quiver(0, 0, 0, *direction, color=color, linewidth=2, arrow_length_ratio=0.16)
        axes.text(*(direction * 1.22), label, color=color, fontsize=11, weight="bold")
    axes.set(xlim=(-0.2, 1.4), ylim=(-0.2, 1.4), zlim=(-0.2, 1.4))
    axes.set_box_aspect((1, 1, 1))
    axes.set_axis_off()


def _render_one(meshes: list[tuple[str, trimesh.Trimesh]], path: Path, title: str,
                combined: bool) -> None:
    """Render shaded oblique anatomy with translucent skeletal reference.

    Source meshes remain in their original RAS+ millimeter coordinates.

    Input:
        meshes: Named meshes and their source geometry.
        path: PNG destination.
        title: Preview name; identifies the selected bone in individual views.
        combined: Whether this is the overview, where every bone is reference.
    Output: None; writes a 1200 by 1200 PNG.
    """
    if not all(np.isfinite(mesh.vertices).all() for _, mesh in meshes):
        raise ValueError("nonfinite_mesh")
    context_present = False
    legend = []
    layers = []
    for name, mesh in meshes:
        if name in _CONTEXT and (combined or name != title):
            color, opacity = _CONTEXT_COLOR, _CONTEXT_ALPHA
            context_present = True
        else:
            color, opacity = _SELECTED_BONE_COLOR, 1.0
            if name not in _CONTEXT:
                color = np.asarray(structure_color(name), dtype=float) / 255.0
            legend.append(Patch(facecolor=color, label=structure_label(name)))
        layers.append((mesh, color, opacity))
    if context_present:
        legend.append(Patch(facecolor=_CONTEXT_COLOR, alpha=_CONTEXT_LEGEND_ALPHA,
                            label="Spine / pelvis reference"))

    figure = plt.figure(figsize=(12, 12), dpi=100, facecolor="white")
    try:
        figure.text(0.06, 0.95, "Anatomy overview" if combined else structure_label(title),
                    fontsize=24, weight="bold", color="#243342")
        figure.text(0.06, 0.919, "Anterior oblique",
                    fontsize=11, color="#5a6773")
        legend_artist = figure.legend(
            handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.047),
            ncol=min(4, len(legend)), frameon=False, fontsize=_LEGEND_FONT_SIZE,
            handlelength=1.2, columnspacing=1.6,
        )
        # Long legends (for example --all-structures) must not cover the anatomy.
        legend_top = legend_artist.get_window_extent().y1 / figure.bbox.height
        bottom = max(_ANATOMY_BOTTOM, legend_top + _LEGEND_GAP)
        side = _ANATOMY_TOP - bottom
        ax = figure.add_axes(((1 - side) / 2, bottom, side, side))
        ax.set_axis_off()
        ax.imshow(_draw_anatomy(layers))
        _orientation(figure)
        reference = ("Translucent bone is a spatial reference" if context_present
                     else "Skeletal reference unavailable")
        caption = f"{reference}; organs may hide one another." if combined else f"{reference}."
        figure.text(0.5, 0.024, caption, ha="center", fontsize=9, color="#68737d")
        figure.savefig(path, format="png", dpi=100, facecolor="white")
    finally:
        plt.close(figure)


def render_previews(scene: trimesh.Scene, output_dir: Path, *, names=None) -> RenderResult:
    """Render individual and combined 1200 by 1200 PNG previews.

    Rendering starts a new process, so a script calling this directly needs the
    usual ``if __name__ == "__main__":`` guard.

    Input:
        scene: Named ``trimesh.Scene`` whose geometry is rendered in place.
        output_dir: Caller-owned preview destination directory.
        names: Optional ordered names that receive their own preview; defaults
            to all scene geometry names. Duplicates and names absent from the
            scene are ignored. Spine/pelvis meshes in the scene are always drawn
            as reference, even when they are not named.
    Output:
        A ``RenderResult`` retaining successful paths and safe failures. A
        non-finite spine/pelvis mesh fails only its own and, when named, the
        combined preview; other views omit it as reference. Previews that cannot
        run because the worker could not be created, crashed, or could not load
        the renderer report ``renderer_unavailable``; others that fail report
        ``preview_failed``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = dict(scene.geometry)
    ordered = tuple(dict.fromkeys(names if names is not None else geometry))
    ordered = tuple(name for name in ordered if name in geometry)
    paths, failures = {}, {}
    context = [name for name, mesh in geometry.items()
               if name in _CONTEXT and np.isfinite(mesh.vertices).all()]
    # Each job contains the output name, structures to show, and whether it is
    # the combined overview. Start with the overview of all selected anatomy.
    combined_names = list(ordered)
    for bone in context:
        if bone not in ordered:
            combined_names.append(bone)
    preview_jobs = [("combined", combined_names, True)]

    for name in ordered:
        # Show this structure alongside the surrounding skeleton.
        individual_names = [name]
        for bone in context:
            # A selected bone is already included above; do not draw it twice.
            if bone != name:
                individual_names.append(bone)
        preview_jobs.append((name, individual_names, False))

    # "spawn" because forking is unsafe for OpenGL and already-threaded libraries.
    spawn = multiprocessing.get_context("spawn")
    try:
        pool = ProcessPoolExecutor(max_workers=1, mp_context=spawn)
    except OSError:
        return RenderResult({}, {name: "renderer_unavailable" for name, _, _ in preview_jobs})
    with pool:
        for name, shown, combined in preview_jobs:
            target = output_dir / f"{name}.png"
            try:
                meshes = [(shown_name, geometry[shown_name]) for shown_name in shown]
                pool.submit(_render_one, meshes, target, name, combined).result()
                paths[name] = target
            except (BrokenProcessPool, ImportError):
                failures[name] = "renderer_unavailable"
            except Exception:
                failures[name] = "preview_failed"
    return RenderResult(paths, failures)


render = render_previews
