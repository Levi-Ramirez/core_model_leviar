"""Render named model meshes as fixed-view PNG previews."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

from core_model.anatomy import structure_color


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


def _bounds(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    """Compute combined finite bounds.

    Input: Meshes to frame.
    Output: ``(minimum, maximum)`` world-coordinate bounds.
    """
    points = np.concatenate([mesh.vertices for mesh in meshes])
    if not np.isfinite(points).all():
        raise ValueError("nonfinite_mesh")
    return points.min(axis=0), points.max(axis=0)


def _set_equal_limits(ax, minimum, maximum) -> None:
    """Set equal physical scale and a shared margin on all axes.

    Input: Matplotlib 3D axes and world-coordinate bounds.
    Output: None; axes limits are updated in place.
    """
    center = (minimum + maximum) / 2
    span = float(np.max(maximum - minimum))
    span = max(span, 1e-6)
    half = span * 0.58
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _render_one(meshes: list[tuple[str, trimesh.Trimesh]], path: Path, title: str,
                combined: bool) -> None:
    """Render one orthographic anterior preview without changing mesh data.

    Input:
        meshes: Named meshes and their source geometry.
        path: PNG destination.
        title: Preview title.
        combined: Whether to include the occlusion limitation caption.
    Output: None; writes a 1200 by 1200 PNG.
    """
    minimum, maximum = _bounds([mesh for _, mesh in meshes])
    figure = plt.figure(figsize=(12, 12), dpi=100, facecolor="white")
    try:
        ax = figure.add_subplot(111, projection="3d")
        ax.set_proj_type("ortho")
        # RAS+: anterior is +Y, so this camera looks from +A toward -A.
        ax.view_init(elev=0, azim=-90)
        for name, mesh in meshes:
            color = np.asarray(structure_color(name), dtype=float) / 255.0
            collection = Poly3DCollection(mesh.triangles, facecolor=(*color, 1.0),
                                           edgecolor="none", linewidth=0)
            ax.add_collection3d(collection)
        _set_equal_limits(ax, minimum, maximum)
        ax.set_axis_off()
        ax.text2D(0.03, 0.96, title, transform=ax.transAxes, fontsize=14,
                  color="black", va="top")
        if combined:
            ax.text2D(0.03, 0.02, "Combined surfaces may occlude one another",
                      transform=ax.transAxes, fontsize=9, color="dimgray")
        # Fixed orientation markers make the view convention explicit.
        ax.text2D(0.05, 0.50, "R", transform=ax.transAxes, fontsize=11)
        ax.text2D(0.95, 0.50, "L", transform=ax.transAxes, fontsize=11, ha="right")
        ax.text2D(0.50, 0.93, "S", transform=ax.transAxes, fontsize=11, ha="center")
        ax.text2D(0.50, 0.07, "I", transform=ax.transAxes, fontsize=11, ha="center")
        figure.savefig(path, format="png", dpi=100, facecolor="white")
    finally:
        plt.close(figure)


def render_previews(scene: trimesh.Scene, output_dir: Path, *, names=None) -> RenderResult:
    """Render individual and combined 1200 by 1200 PNG previews.

    Input:
        scene: Named ``trimesh.Scene`` whose geometry is rendered in place.
        output_dir: Caller-owned preview destination directory.
        names: Optional ordered names; defaults to all scene geometry names.
    Output:
        A ``RenderResult`` retaining successful paths and safe failures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = dict(scene.geometry)
    ordered = tuple(names) if names is not None else tuple(geometry)
    paths, failures = {}, {}
    selected = [(name, geometry[name]) for name in ordered if name in geometry]
    for name, meshes, combined in [
        ("combined", selected, True),
        *[(name, [(name, geometry[name])], False) for name in ordered if name in geometry],
    ]:
        target = output_dir / f"{name}.png"
        try:
            _render_one(meshes, target, name, combined)
            paths[name] = target
        except Exception:
            failures[name] = "preview_failed"
    return RenderResult(paths, failures)


render = render_previews
