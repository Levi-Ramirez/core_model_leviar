"""Shared values and allowed choices for a segmentation run."""

from typing import Literal, get_args

Stage = Literal["generation", "ingest", "segmentation", "mesh", "model", "render"]
Device = Literal["cpu", "gpu", "mps"]
Quality = Literal["standard", "fast"]
SupportStatus = Literal["unverified", "unsupported"]
StructureStatus = Literal["included", "skipped", "failed"]
ManifestStatus = Literal["running", "complete", "partial", "failed"]
ResultStatus = Literal["complete", "partial"]
SpatialUnits = Literal["mm"]
ArtifactKind = Literal[
    "combined_mesh", "structure_mesh", "combined_preview", "structure_preview"
]

# Derive runtime choices from type aliases so validation cannot drift from typing.
STAGES = get_args(Stage)
DEVICES = get_args(Device)
QUALITIES = get_args(Quality)
SUPPORT_STATUSES = get_args(SupportStatus)
STRUCTURE_STATUSES = get_args(StructureStatus)
MANIFEST_STATUSES = get_args(ManifestStatus)
RESULT_STATUSES = get_args(ResultStatus)
SPATIAL_UNITS = get_args(SpatialUnits)
ARTIFACT_KINDS = get_args(ArtifactKind)

DEFAULT_DEVICE: Device = "cpu"
DEFAULT_QUALITY: Quality = "standard"
DEFAULT_SPATIAL_UNITS: SpatialUnits = "mm"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
SEGMENTATION_TASK = "total"
