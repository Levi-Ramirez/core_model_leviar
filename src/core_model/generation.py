"""Generate a 3D model and manage its owned workspace and artifacts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Callable

from core_model.anatomy import resolve_selection
from core_model.constants.contracts_constants import MANIFEST_FILENAME
from core_model.contracts import (
    Artifact, GenerationError, GenerationManifest, GenerationOptions,
    GenerationResult, StageIssue,
)
from core_model.ingest import ingest
from core_model.mesh import MeshError, extract_mesh
from core_model.model import ModelError, assemble_model
from core_model.render import render_previews
from core_model.segmentation import segment


def _write_manifest(output_dir: Path, manifest: GenerationManifest) -> None:
    """Write the current manifest.

    Input: Output directory and manifest record.
    Output: None; replaces the manifest file with validated schema-v1 JSON.
    """
    target = output_dir / MANIFEST_FILENAME
    temporary = output_dir / f".{MANIFEST_FILENAME}.tmp"
    temporary.write_text(manifest.to_json(), encoding="utf-8")
    temporary.replace(target)


def _fail(output_dir: Path, manifest: GenerationManifest, stage: str, reason: str):
    """Record a safe fatal stage error and raise a generation exception.

    Input: Output directory, current manifest, failed stage, and reason code.
    Output: Never returns; writes a failed manifest and raises ``GenerationError``.
    """
    manifest.status = "failed"
    manifest.errors.append(StageIssue(stage, reason))
    _write_manifest(output_dir, manifest)
    raise GenerationError(stage, reason, manifest_path=output_dir / MANIFEST_FILENAME)


def generate_model(input_path: str | Path, output_dir: str | Path,
                   options: GenerationOptions | None = None,
                   progress: Callable[[str, str], None] | None = None) -> GenerationResult:
    """Generate meshes and previews from one local scan.

    Input:
        input_path: DICOM directory/zip or validated NIfTI input.
        output_dir: New or empty caller-owned directory for final artifacts.
        options: Generation settings, defaulting to CPU standard quality.
        progress: Optional callback receiving ``(stage, state)`` updates.
    Output:
        A complete or partial ``GenerationResult``; fatal stage errors raise.
    """
    options = options or GenerationOptions()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GenerationError("generation", "output_directory_not_empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = GenerationManifest(options=options, resolved_device=options.device)
    _write_manifest(output_dir, manifest)
    system_timings = {}
    def notify(stage, state):
        if progress is not None:
            progress(stage, state)
    try:
        with TemporaryDirectory(prefix="workspace-", dir=output_dir) as workspace_name:
            workspace = Path(workspace_name)
            started = time.monotonic()
            notify("ingest", "started")
            volume = ingest(input_path, workspace, series_id=options.series_id)
            system_timings["ingest"] = time.monotonic() - started
            notify("ingest", "completed")
            manifest.source_geometry = volume.geometry
            _write_manifest(output_dir, manifest)

            selection = resolve_selection(options)
            started = time.monotonic()
            (workspace / "segmentation").mkdir()
            notify("segmentation", "started (first run may download model weights)")
            segmentation = segment(
                volume.nifti_path, workspace / "segmentation", selection.structures,
                device=options.device, quality=options.quality,
            )
            system_timings["segmentation"] = time.monotonic() - started
            notify("segmentation", "completed")
            _write_manifest(output_dir, manifest)

            meshes, empty = {}, set()
            started = time.monotonic()
            notify("mesh", "started")
            for name, path in segmentation.masks.items():
                try:
                    meshes[name] = extract_mesh(
                        path, expected_shape=volume.geometry.shape,
                        expected_affine=volume.geometry.affine,
                    )
                except MeshError as error:
                    if error.reason == "empty_mask":
                        empty.add(name)
                    else:
                        raise GenerationError("mesh", error.reason) from None
            system_timings["mesh"] = time.monotonic() - started
            notify("mesh", "completed")
            notify("model", "started")
            try:
                model = assemble_model(
                    selection, meshes, output_dir, empty_structures=frozenset(empty),
                    explicit_interests=options.include is not None,
                )
            except ModelError as error:
                raise GenerationError("model", error.reason) from None
            system_timings["model"] = time.monotonic() - started
            notify("model", "completed")

            started = time.monotonic()
            notify("render", "started")
            previews = render_previews(model.scene, output_dir / "previews", names=selection.structures)
            system_timings["render"] = time.monotonic() - started
            notify("render", "completed")
            artifacts = list(model.artifacts)
            for name, path in previews.paths.items():
                if name == "combined":
                    artifacts.append(Artifact("previews/combined.png", "combined_preview"))
                else:
                    artifacts.append(Artifact(f"previews/{name}.png", "structure_preview", name))
            warnings = [StageIssue("model", warning.split(":", 1)[0],
                                   warning.split(":", 1)[1] if ":" in warning else None)
                        for warning in model.warnings]
            errors = [StageIssue("render", reason,
                                 name if name != "combined" else None)
                      for name, reason in previews.failures.items()]
            manifest.status = "partial" if errors else "complete"
            manifest.artifacts = artifacts
            manifest.structures = list(model.structures)
            manifest.warnings = warnings
            manifest.errors = errors
            manifest.stage_timings = system_timings
            _write_manifest(output_dir, manifest)
            return GenerationResult(output_dir, manifest.status, artifacts,
                                    list(model.structures), warnings, errors, system_timings)
    except GenerationError as error:
        _fail(output_dir, manifest, error.stage, error.reason)
    except Exception:
        _fail(output_dir, manifest, "generation", "generation_failed")


run = generate_model
