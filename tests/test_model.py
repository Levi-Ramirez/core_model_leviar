import numpy as np
import trimesh
import pytest

from core_model.anatomy import AnatomySelection
from core_model.mesh import MeshResult
from core_model.model import ModelError, assemble_model


def cube(offset):
    mesh = trimesh.creation.box(extents=(2, 2, 2))
    mesh.apply_translation(offset)
    return mesh


def test_assemble_model_exports_named_and_combined_binary_ply(tmp_path):
    selection = AnatomySelection(("liver",), ("sacrum",))
    source = cube((1, 2, 3))
    result = assemble_model(
        selection,
        {"liver": MeshResult(source, False)},
        tmp_path,
    )

    assert set(result.scene.geometry) == {"liver"}
    assert (tmp_path / "model.ply").read_bytes().startswith(b"ply\n")
    assert (tmp_path / "structures/liver.ply").exists()
    assert [outcome.name for outcome in result.structures] == ["liver", "sacrum"]
    assert result.structures[0].status == "included"
    assert result.structures[1].reason == "missing_context"
    round_trip = trimesh.load(tmp_path / "structures/liver.ply", force="mesh")
    assert len(round_trip.vertices) == len(source.vertices)
    assert np.asarray(round_trip.visual.vertex_colors)[:, :3].tolist() == [[180, 70, 60]] * 8


def test_empty_default_and_context_masks_are_skipped_with_warnings(tmp_path):
    selection = AnatomySelection(("liver",), ("sacrum",))
    result = assemble_model(
        selection,
        {"liver": cube((0, 0, 0))},
        tmp_path,
        empty_structures=frozenset({"sacrum"}),
    )
    assert "missing_context:sacrum" in result.warnings
    assert result.structures[1].reason == "empty_mask"


def test_empty_explicit_interest_is_fatal(tmp_path):
    selection = AnatomySelection(("liver",), ())
    with pytest.raises(ModelError, match="requested_structure_empty"):
        assemble_model(selection, {}, tmp_path,
                       empty_structures=frozenset({"liver"}), explicit_interests=True)


def test_no_interest_mesh_is_fatal_even_when_context_exists(tmp_path):
    selection = AnatomySelection(("liver",), ("sacrum",))
    with pytest.raises(ModelError, match="no_anatomy_of_interest"):
        assemble_model(selection, {"sacrum": cube((0, 0, 0))}, tmp_path,
                       empty_structures=frozenset({"liver"}))
