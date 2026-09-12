"""Shared pipeline records, independent of inference, imaging, and CLI code.

Manifest records contain only deliberate provenance fields. Callers must pass
non-identifying class/series identifiers and software/model metadata, never raw
DICOM tags or exception messages. These records are not a general PHI scrubber.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from core_model.constants.contracts_constants import (
    ARTIFACT_KINDS, DEFAULT_DEVICE, DEFAULT_QUALITY, DEFAULT_SPATIAL_UNITS,
    DEVICES, MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION, MANIFEST_STATUSES,
    QUALITIES, RESULT_STATUSES, SEGMENTATION_TASK, SPATIAL_UNITS, STAGES,
    STRUCTURE_STATUSES, SUPPORT_STATUSES,
    ArtifactKind, Device, ManifestStatus, Quality, ResultStatus, SpatialUnits,
    Stage, StructureStatus, SupportStatus,
)
from core_model.utils.validation import (
    relative_artifact_path, validate_choice, validate_identifier,
)


@dataclass(frozen=True)
class GenerationOptions:
    series_id: str | None = None
    include: tuple[str, ...] | None = None
    all_structures: bool = False
    device: Device = DEFAULT_DEVICE
    quality: Quality = DEFAULT_QUALITY

    def __post_init__(self):
        validate_choice(self.device, DEVICES, "device")
        validate_choice(self.quality, QUALITIES, "quality")
        if not isinstance(self.all_structures, bool):
            raise ValueError("all_structures must be boolean")
        if self.series_id is not None:
            validate_identifier(self.series_id)
        if self.include is not None:
            if isinstance(self.include, str) or not self.include:
                raise ValueError("include must be a nonempty sequence of class names")
            if self.all_structures:
                raise ValueError("include and all_structures are mutually exclusive")
            for name in self.include:
                validate_identifier(name)
            # Snapshot caller-owned lists and preserve order while deduplicating.
            object.__setattr__(self, "include", tuple(dict.fromkeys(self.include)))


@dataclass(frozen=True)
class SeriesCandidate:
    series_id: str
    modality: str
    slice_count: int
    support_status: SupportStatus = "unverified"
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self):
        validate_identifier(self.series_id)
        validate_identifier(self.modality)
        if type(self.slice_count) is not int or self.slice_count < 0:
            raise ValueError("slice_count must be a nonnegative integer")
        validate_choice(self.support_status, SUPPORT_STATUSES, "support status")
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))
        for reason in self.rejection_reasons:
            validate_identifier(reason)
        if (self.support_status == "unsupported") != bool(self.rejection_reasons):
            raise ValueError("Only unsupported candidates require rejection reasons")


@dataclass(frozen=True)
class StageIssue:
    """A stable reason code; user-facing explanations belong at presentation time."""

    stage: Stage
    reason: str
    structure: str | None = None

    def __post_init__(self):
        validate_choice(self.stage, STAGES, "stage")
        validate_identifier(self.reason)
        if self.structure is not None:
            validate_identifier(self.structure)


@dataclass(frozen=True)
class ImageGeometry:
    """Normalized image geometry; detailed affine/grid validation lives in ingest."""

    shape: tuple[int, int, int]
    affine: tuple[tuple[float, ...], ...]
    units: SpatialUnits = DEFAULT_SPATIAL_UNITS

    def __post_init__(self):
        shape = tuple(self.shape)
        affine = tuple(tuple(float(v) for v in row) for row in self.affine)
        if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("Geometry requires three positive integer dimensions")
        if len(affine) != 4 or any(len(row) != 4 for row in affine):
            raise ValueError("Geometry requires a 4 by 4 affine")
        if not all(math.isfinite(v) for row in affine for v in row):
            raise ValueError("Affine values must be finite")
        validate_choice(self.units, SPATIAL_UNITS, "spatial units")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "affine", affine)


@dataclass(frozen=True)
class StructureOutcome:
    name: str
    status: StructureStatus
    reason: str | None = None
    clipped: bool = False
    color: tuple[int, int, int] | None = None
    geometry: ImageGeometry | None = None

    def __post_init__(self):
        validate_identifier(self.name)
        if self.geometry is not None and type(self.geometry) is not ImageGeometry:
            raise ValueError("Mask geometry must be an ImageGeometry record")
        validate_choice(self.status, STRUCTURE_STATUSES, "structure status")
        if (self.status != "included") != (self.reason is not None):
            raise ValueError("Skipped and failed structures require a reason")
        if self.reason is not None:
            validate_identifier(self.reason)
        if self.color is not None:
            color = tuple(self.color)
            if len(color) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in color):
                raise ValueError("Color must contain three integer RGB channels (0–255)")
            object.__setattr__(self, "color", color)


@dataclass(frozen=True)
class Artifact:
    """One successfully written file, relative to the output directory."""

    path: str
    kind: ArtifactKind
    structure: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "path", relative_artifact_path(self.path))
        validate_choice(self.kind, ARTIFACT_KINDS, "artifact kind")
        if self.kind.startswith("structure_") != (self.structure is not None):
            raise ValueError("Only per-structure artifacts require a structure name")
        if self.structure is not None:
            validate_identifier(self.structure)


def _validate_records(artifacts, structures, timings):
    if any(type(a) is not Artifact for a in artifacts):
        raise ValueError("Artifacts must be Artifact records")
    if any(type(s) is not StructureOutcome for s in structures):
        raise ValueError("Structures must be StructureOutcome records")
    if len({a.path for a in artifacts}) != len(artifacts):
        raise ValueError("Artifact paths must be unique")
    if len({s.name for s in structures}) != len(structures):
        raise ValueError("Structure outcomes must be unique by name")
    for stage, seconds in timings.items():
        validate_choice(stage, STAGES, "timing stage")
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Stage timings must be finite nonnegative seconds")


def _validate_issues(warnings, errors):
    if any(type(issue) is not StageIssue for issue in (*warnings, *errors)):
        raise ValueError("Warnings and errors must be StageIssue records, not raw exceptions")


def _validate_finished_status(status, structures, errors):
    if status not in RESULT_STATUSES:
        return
    if any(s.status == "failed" for s in structures):
        raise ValueError("Failed structures are fatal, not a completed or partial result")
    if status == "partial" and any(e.stage != "render" for e in errors):
        raise ValueError("Only preview failures can produce a partial result")


@dataclass
class GenerationManifest:
    """Schema v1 run provenance. Only completed artifacts belong in artifacts.

    Fields may be absent while a run is starting or after an early failure.
    Filesystem writes and stage transitions are the pipeline's responsibility.
    """

    options: GenerationOptions = field(default_factory=GenerationOptions)
    status: ManifestStatus = "running"
    resolved_device: Device | None = None
    software_versions: dict[str, str] = field(default_factory=dict)
    weight_identifier: str | None = None
    source_geometry: ImageGeometry | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    structures: list[StructureOutcome] = field(default_factory=list)
    warnings: list[StageIssue] = field(default_factory=list)
    errors: list[StageIssue] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        if type(self.options) is not GenerationOptions:
            raise ValueError("Options must be a GenerationOptions record")
        if self.source_geometry is not None and type(self.source_geometry) is not ImageGeometry:
            raise ValueError("Source geometry must be an ImageGeometry record")
        validate_choice(self.status, MANIFEST_STATUSES, "manifest status")
        if self.resolved_device is not None:
            validate_choice(self.resolved_device, DEVICES, "resolved device")
        _validate_records(self.artifacts, self.structures, self.stage_timings)
        _validate_issues(self.warnings, self.errors)
        _validate_finished_status(self.status, self.structures, self.errors)
        if self.status in ("partial", "failed") and not self.errors:
            raise ValueError("Partial and failed manifests require error records")
        if self.status == "complete" and self.errors:
            raise ValueError("A complete manifest cannot contain errors")
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "task": SEGMENTATION_TASK, **asdict(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass
class GenerationResult:
    """Returned to callers; local output paths are not manifest provenance."""

    output_dir: Path
    status: ResultStatus
    artifacts: list[Artifact] = field(default_factory=list)
    structures: list[StructureOutcome] = field(default_factory=list)
    warnings: list[StageIssue] = field(default_factory=list)
    errors: list[StageIssue] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        validate_choice(self.status, RESULT_STATUSES, "result status")
        _validate_records(self.artifacts, self.structures, self.stage_timings)
        _validate_issues(self.warnings, self.errors)
        _validate_finished_status(self.status, self.structures, self.errors)
        if (self.status == "partial") != bool(self.errors):
            raise ValueError("Only partial results require error records")

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / MANIFEST_FILENAME

    @property
    def included_classes(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.structures if s.status == "included")

    @property
    def skipped_classes(self) -> dict[str, str]:
        return {s.name: s.reason for s in self.structures if s.status == "skipped"}

    def to_dict(self) -> dict:
        """Serialize a caller-facing result, including its local output directory."""
        self.__post_init__()
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["manifest_path"] = str(self.manifest_path)
        return data


class GenerationError(Exception):
    """Fatal stage failure with a safe reason code and optional local results.

    Do not pass upstream exception text as reason. Exception chaining may be
    used internally, but CLI/logging must display only the safe issue record.
    """

    def __init__(self, stage: Stage, reason: str, *, manifest_path: Path | None = None,
                 result: GenerationResult | None = None):
        self.issue = StageIssue(stage, reason)
        self.stage = stage
        self.reason = reason
        self.result = result
        self.manifest_path = Path(manifest_path) if manifest_path is not None else (
            result.manifest_path if result is not None else None
        )
        super().__init__(f"{stage}: {reason}")
