"""Assemble named meshes and write the PLY artifacts for a model run."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import trimesh

from core_model.anatomy import AnatomySelection, structure_color
from core_model.contracts import Artifact, StructureOutcome
from core_model.mesh import MeshResult


class ModelError(Exception):
    """Fatal model assembly or export error with a safe reason code.

    Input:
        reason: Stable, non-identifying model error code.
    Output:
        An exception carrying the model failure reason.
    """

    def __init__(self, reason: str):
        super().__init__(f"model: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class ModelResult:
    """Named scene, structure outcomes, and successfully written artifacts.

    Input:
        scene: Named ``trimesh.Scene`` containing nonempty structures.
        structures: One outcome for every selected interest/context name.
        artifacts: Relative PLY artifact records.
        warnings: Safe model warnings, such as missing context.
    Output:
        An immutable model-stage result.
    """

    scene: trimesh.Scene
    structures: tuple[StructureOutcome, ...]
    artifacts: tuple[Artifact, ...]
    warnings: tuple[str, ...] = ()


def _mesh_from_value(value) -> trimesh.Trimesh:
    """Return a mesh from a mesh-stage result or direct trimesh value.

    Input: A ``MeshResult`` or ``trimesh.Trimesh`` value.
    Output: The underlying mesh, or raises ``ModelError`` for invalid values.
    """
    mesh = value.mesh if isinstance(value, MeshResult) else value
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ModelError("invalid_mesh")
    return mesh


def _colored_copy(mesh: trimesh.Trimesh, color: tuple[int, int, int]) -> trimesh.Trimesh:
    """Copy a mesh and attach stable opaque RGBA vertex colors.

    Input:
        mesh: Source mesh whose geometry must remain unchanged.
        color: Stable RGB structure color.
    Output: A copied colored mesh for scene/export use.
    """
    copy = mesh.copy()
    copy.visual.vertex_colors = tuple(color) + (255,)
    return copy


def assemble_model(selection: AnatomySelection, meshes: Mapping[str, object], output_dir: Path,
                   *, empty_structures: frozenset[str] = frozenset(),
                   explicit_interests: bool = False) -> ModelResult:
    """Assemble selected structures and export individual and combined PLY files.

    Input:
        selection: Interest and spine/pelvis context names to assemble.
        meshes: Mapping of available names to ``MeshResult`` or ``Trimesh``.
        output_dir: Caller-owned output directory for ``model.ply`` and files.
        empty_structures: Names with valid but empty masks.
        explicit_interests: Whether missing/empty interest masks are fatal.
    Output:
        A ``ModelResult`` with a named scene, outcomes, and artifact records.
    """
    output_dir = Path(output_dir)
    structures = selection.structures
    outcomes = []
    warnings = []
    colored = {}
    for name in structures:
        is_interest = name in selection.interests
        if name in empty_structures:
            if is_interest and explicit_interests:
                raise ModelError("requested_structure_empty")
            reason = "empty_mask"
            outcomes.append(StructureOutcome(name, "skipped", reason, color=structure_color(name)))
            if name in selection.context:
                warnings.append(f"missing_context:{name}")
            continue
        if name not in meshes:
            if is_interest:
                raise ModelError("missing_requested_structure")
            outcomes.append(StructureOutcome(name, "skipped", "missing_context", color=structure_color(name)))
            warnings.append(f"missing_context:{name}")
            continue
        value = meshes[name]
        mesh = _mesh_from_value(value)
        color = structure_color(name)
        colored[name] = _colored_copy(mesh, color)
        clipped = bool(value.clipped) if isinstance(value, MeshResult) else False
        outcomes.append(StructureOutcome(name, "included", clipped=clipped, color=color))

    interest_names = {name for name in selection.interests if name in colored}
    if not interest_names:
        raise ModelError("no_anatomy_of_interest")
    if not any(name in colored for name in selection.context):
        warnings.append("all_context_missing")
    if not colored:
        raise ModelError("empty_scene")

    scene = trimesh.Scene()
    for name in structures:
        if name in colored:
            scene.add_geometry(colored[name], node_name=name, geom_name=name)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_dir = output_dir / "structures"
        structure_dir.mkdir(exist_ok=True)
        artifacts = []
        for name, mesh in colored.items():
            path = structure_dir / f"{name}.ply"
            mesh.export(path, file_type="ply")
            artifacts.append(Artifact(f"structures/{name}.ply", "structure_mesh", name))
        combined = trimesh.util.concatenate(tuple(colored.values()))
        combined.export(output_dir / "model.ply", file_type="ply")
        artifacts.insert(0, Artifact("model.ply", "combined_mesh"))
    except Exception:
        raise ModelError("export_failed") from None
    return ModelResult(scene, tuple(outcomes), tuple(artifacts), tuple(dict.fromkeys(warnings)))


build_model = assemble_model
