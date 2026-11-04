"""Render named model meshes as fixed-view PNG previews."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

from core_model.anatomy import structure_color, structure_label
from core_model.constants.anatomy_constants import SPINE_AND_PELVIS_STRUCTURES


_CONTEXT = frozenset(SPINE_AND_PELVIS_STRUCTURES)
# RAS+: +Y is anterior, +X is right, +Z is superior.
_ELEVATION, _AZIMUTH = 18, 65
_LIGHT = LightSource(azdeg=25, altdeg=50)
_CONTEXT_COLOR = np.array((0.76, 0.75, 0.71))
_CONTEXT_ALPHA = 0.20
# A mesh stacks front and back faces, so its reference gray looks denser than a
# single legend swatch drawn at the same alpha.
_CONTEXT_LEGEND_ALPHA = 0.4
_SELECTED_BONE_COLOR = np.array((135, 206, 250)) / 255.0  # Light sky blue.

# Numerical floor for degenerate bounds; mesh coordinates are in millimeters.
_MIN_AXIS_EXTENT_MM = 1e-6
# Keep the viewing box from collapsing around thin or nearly planar structures.
_MIN_AXIS_FRACTION_OF_LONGEST = 0.08
# Leave 2% of the framed span as margin on each side.
_FRAME_PADDING_FACTOR = 1.04
# Half of the 3D view's visible span in box units; Matplotlib's axes show
# -0.95 to +0.9 horizontally and vertically, so the smaller side is the limit.
_VIEW_HALF_SPAN = 0.9
# Default vertical span of the anatomy axes, as figure fractions.
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


def _finite_points(meshes: list[trimesh.Trimesh]) -> np.ndarray:
    """Collect the vertices used to frame a view.

    Input: Meshes to frame.
    Output: An ``(n, 3)`` array of finite world coordinates.
    """
    points = np.concatenate([mesh.vertices for mesh in meshes])
    if not np.isfinite(points).all():
        raise ValueError("nonfinite_mesh")
    return points


def _screen_axes() -> np.ndarray:
    """Return the camera's screen directions in world coordinates.

    Input: None; uses the fixed preview camera.
    Output: A ``(2, 3)`` array of unit vectors for screen right and screen up.
    """
    elevation, azimuth = np.radians((_ELEVATION, _AZIMUTH))
    return np.array((
        (-np.sin(azimuth), np.cos(azimuth), 0),
        (-np.sin(elevation) * np.cos(azimuth), -np.sin(elevation) * np.sin(azimuth),
         np.cos(elevation)),
    ))


def _frame_view(ax, points: np.ndarray) -> None:
    """Fit the camera to the visible outline at equal physical scale.

    The 3D bounding box's corners are usually empty, so sizing from them leaves
    whitespace around the anatomy. Scale and center on the projected vertices.

    Input: Matplotlib 3D axes and the finite world coordinates to show.
    Output: None; axes limits and box aspect are updated in place.
    """
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    extent = np.maximum(maximum - minimum, _MIN_AXIS_EXTENT_MM)
    extent = np.maximum(extent, extent.max() * _MIN_AXIS_FRACTION_OF_LONGEST)
    extent *= _FRAME_PADDING_FACTOR

    screen_axes = _screen_axes()
    screen = points @ screen_axes.T
    screen_low, screen_high = screen.min(axis=0), screen.max(axis=0)
    # Limits may sit off the bounds' center; only depth is left unchanged.
    center = (minimum + maximum) / 2
    center += ((screen_low + screen_high) / 2 - screen_axes @ center) @ screen_axes
    for setter, mid, span in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, extent):
        setter(mid - span / 2, mid + span / 2)

    # Match display-box proportions to the axis ranges so one millimeter has
    # the same scale on every axis, without requiring a cubic viewing box.
    ax.set_box_aspect(tuple(extent))
    box_units_per_mm = ax.get_box_aspect()[0] / extent[0]
    outline = max((screen_high - screen_low).max(), _MIN_AXIS_EXTENT_MM) * _FRAME_PADDING_FACTOR
    ax.set_box_aspect(tuple(extent), zoom=2 * _VIEW_HALF_SPAN / (box_units_per_mm * outline))


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
    Matplotlib transparency/depth sorting is illustrative, not volumetric.

    Input:
        meshes: Named meshes and their source geometry.
        path: PNG destination.
        title: Preview name; identifies the selected bone in individual views.
        combined: Whether this is the overview, where every bone is reference.
    Output: None; writes a 1200 by 1200 PNG.
    """
    points = _finite_points([mesh for _, mesh in meshes])
    figure = plt.figure(figsize=(12, 12), dpi=100, facecolor="white")
    try:
        ax = figure.add_axes((0.02, _ANATOMY_BOTTOM, 0.96, _ANATOMY_TOP - _ANATOMY_BOTTOM),
                             projection="3d")
        ax.set_proj_type("ortho")
        ax.view_init(elev=_ELEVATION, azim=_AZIMUTH)
        context_present = False
        legend = []
        for name, mesh in meshes:
            if name in _CONTEXT and (combined or name != title):
                color, alpha = _CONTEXT_COLOR, _CONTEXT_ALPHA
                context_present = True
            else:
                color, alpha = _SELECTED_BONE_COLOR, 1.0
                if name not in _CONTEXT:
                    color = np.asarray(structure_color(name), dtype=float) / 255.0
                legend.append(Patch(facecolor=color, label=structure_label(name)))
            collection = Poly3DCollection(
                mesh.triangles, facecolors=(*color, alpha), edgecolor="none", linewidth=0,
                shade=True, lightsource=_LIGHT,
            )
            ax.add_collection3d(collection)
        _frame_view(ax, points)
        ax.set_axis_off()
        figure.text(0.06, 0.95, "Anatomy overview" if combined else structure_label(title),
                    fontsize=24, weight="bold", color="#243342")
        figure.text(0.06, 0.919, "Anterior oblique",
                    fontsize=11, color="#5a6773")
        _orientation(figure)
        if context_present:
            legend.append(Patch(facecolor=_CONTEXT_COLOR, alpha=_CONTEXT_LEGEND_ALPHA,
                                label="Spine / pelvis reference"))
        legend_artist = figure.legend(
            handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.047),
            ncol=min(4, len(legend)), frameon=False, fontsize=_LEGEND_FONT_SIZE,
            handlelength=1.2, columnspacing=1.6,
        )
        # Long legends (for example --all-structures) must not cover the anatomy.
        legend_top = legend_artist.get_window_extent().y1 / figure.bbox.height
        bottom = max(_ANATOMY_BOTTOM, legend_top + _LEGEND_GAP)
        ax.set_position((0.02, bottom, 0.96, _ANATOMY_TOP - bottom))
        if context_present:
            caption = "Translucent bone is a spatial reference; surface overlap is approximate."
        elif combined:
            caption = "Skeletal reference unavailable; surface overlap is approximate."
        else:
            caption = "Skeletal reference unavailable in this view."
        figure.text(0.5, 0.024, caption, ha="center", fontsize=9, color="#68737d")
        figure.savefig(path, format="png", dpi=100, facecolor="white")
    finally:
        plt.close(figure)


def render_previews(scene: trimesh.Scene, output_dir: Path, *, names=None) -> RenderResult:
    """Render individual and combined 1200 by 1200 PNG previews.

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
        combined preview; other views omit it as reference.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = dict(scene.geometry)
    ordered = tuple(dict.fromkeys(names if names is not None else geometry))
    ordered = tuple(name for name in ordered if name in geometry)
    paths, failures = {}, {}
    selected = [(name, geometry[name]) for name in ordered]
    context = [(name, mesh) for name, mesh in geometry.items()
               if name in _CONTEXT and np.isfinite(mesh.vertices).all()]
    selected.extend((name, mesh) for name, mesh in context if name not in ordered)
    # Each job contains the output name, meshes to show, and whether it is
    # the combined overview. Start with the overview of all selected anatomy.
    preview_jobs = [("combined", selected, True)]

    for name in ordered:
        # Show this structure alongside the surrounding skeleton.
        individual_meshes = [(name, geometry[name])]
        for bone_name, bone_mesh in context:
            # A selected bone is already included above; do not draw it twice.
            if bone_name != name:
                individual_meshes.append((bone_name, bone_mesh))

        preview_jobs.append((name, individual_meshes, False))

    for name, meshes, combined in preview_jobs:
        target = output_dir / f"{name}.png"
        try:
            _render_one(meshes, target, name, combined)
            paths[name] = target
        except Exception:
            failures[name] = "preview_failed"
    return RenderResult(paths, failures)


render = render_previews
