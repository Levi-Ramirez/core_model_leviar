from pathlib import Path

import trimesh

import core_model.generation as pipeline
from core_model.contracts import GenerationOptions, ImageGeometry
from core_model.ingest import IngestedVolume
from core_model.mesh import MeshResult
from core_model.render import RenderResult
from core_model.segmentation import SegmentationResult


def test_generate_model_wires_stages_and_writes_manifest(tmp_path, monkeypatch):
    input_path = tmp_path / "input.nii.gz"
    input_path.write_bytes(b"synthetic")
    output = tmp_path / "output"
    geometry = ImageGeometry((4, 4, 4), tuple(tuple(float(v) for v in row) for row in __import__("numpy").eye(4)))
    source = tmp_path / "volume.nii.gz"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(pipeline, "ingest", lambda *a, **k: IngestedVolume(source, geometry))
    monkeypatch.setattr(pipeline, "segment", lambda *a, **k: SegmentationResult({}, Path(a[1]), "cpu", "standard"))
    monkeypatch.setattr(pipeline, "resolve_selection", lambda options: __import__("core_model.anatomy", fromlist=["AnatomySelection"]).AnatomySelection(("liver",), ()))
    mesh = MeshResult(trimesh.creation.box(), False)
    monkeypatch.setattr(pipeline, "extract_mesh", lambda *a, **k: mesh)
    monkeypatch.setattr(pipeline, "assemble_model", lambda selection, meshes, output_dir, **kwargs: __import__("core_model.model", fromlist=["assemble_model"]).assemble_model(selection, {"liver": mesh}, output_dir, **kwargs))
    rendered = {}

    def fake_render(scene, preview_dir, *, names):
        rendered.update(names=names, geometry=set(scene.geometry))
        path = Path(preview_dir) / "combined.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic")
        return RenderResult({"combined": path}, {})

    monkeypatch.setattr(pipeline, "render_previews", fake_render)
    result = pipeline.generate_model(input_path, output, GenerationOptions(include=("liver",)))
    assert result.status == "complete"
    assert rendered == {"names": ("liver",), "geometry": {"liver"}}
    assert result.manifest_path.exists()
    assert (output / "model.ply").exists()
    assert (output / "previews/combined.png").exists()


def test_generate_model_rejects_nonempty_output_before_work(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("keep")
    try:
        pipeline.generate_model(tmp_path / "input.nii.gz", output)
    except Exception as error:
        assert getattr(error, "reason", None) == "output_directory_not_empty"
    else:
        raise AssertionError("nonempty output was accepted")
