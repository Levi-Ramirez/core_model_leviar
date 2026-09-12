"""Discover DICOM series and ingest one CT volume into a caller-owned workspace."""

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory, mkdtemp
import warnings

import dicom2nifti
from dicom2nifti import settings
from dicom2nifti.exceptions import ConversionError, ConversionValidationError
import pydicom
from pydicom.uid import CTImageStorage, MediaStorageDirectoryStorage, UID

from core_model.constants.ingest_constants import (
    CONVERSION_REASONS, REQUIRED_CONVERTER_VALIDATIONS,
)
from core_model.contracts import GenerationError, ImageGeometry, SeriesCandidate
from core_model.utils.archive import extract_scan_zip
from core_model.utils.nifti import normalize_nifti

logger = logging.getLogger(__name__)


class IngestError(GenerationError):
    """Safe input error with candidate records for CLI/library selection.

    Input:
        reason: Stable, non-identifying ingestion error code.
        candidates: Optional series candidates that can be selected.
    Output:
        An exception carrying the safe reason and candidate records.
    """

    def __init__(self, reason: str, candidates=()):
        super().__init__("ingest", reason)
        self.candidates = tuple(candidates)

    @property
    def selection_commands(self) -> tuple[str, ...]:
        """Build local shell templates for supported candidate series.

        Input: The candidate records stored on this error.
        Output: Commands containing ``--series`` IDs and local shell variables;
            no patient paths or DICOM metadata.
        """
        return tuple(
            'core-model generate "$SCAN_INPUT" --output "$MODEL_OUTPUT" '
            f'--series {candidate.series_id}'
            for candidate in self.candidates if candidate.support_status == "unverified"
        )


@dataclass(frozen=True)
class IngestedVolume:
    nifti_path: Path
    geometry: ImageGeometry
    selected_series: SeriesCandidate | None = None


@dataclass
class _Series:
    candidate: SeriesCandidate
    files: list[Path] = field(repr=False)


@dataclass(frozen=True)
class _DicomMetadata:
    """Metadata extracted from one readable DICOM image."""

    study_uid: str
    series_uid: str
    modality: str
    sop_class_uid: str
    is_multiframe: bool


def _series_id(study_uid: str, series_uid: str) -> str:
    """Create a stable, non-identifying ID for a study/series pair.

    Input:
        study_uid: Internal DICOM StudyInstanceUID.
        series_uid: Internal DICOM SeriesInstanceUID.
    Output: A collision-checkable ``series_<digest>`` identifier.
    """
    digest = hashlib.sha256(f"{study_uid}\0{series_uid}".encode("ascii")).hexdigest()
    return f"series_{digest[:16]}"


def extract_dicom_meta_data(path: Path) -> _DicomMetadata | None:
    """Extract discovery metadata from one DICOM file without loading pixels.

    Input: A path to one candidate DICOM file.
    Output: A ``_DicomMetadata`` record, or ``None`` for a directory or
        recognized DICOMDIR index.
    Malformed or unrecognized files raise a safe input error. Raw IDs are
    internal grouping data and must not be logged or displayed.
    """
    if path.is_dir():
        return None
    try:
        # Do not emit malformed tag values through pydicom's warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            # DICOMDIR is an index, not a series or a malformed image.
            if ds.file_meta.get("MediaStorageSOPClassUID") == MediaStorageDirectoryStorage:
                return None
    except pydicom.errors.InvalidDicomError:
        raise IngestError("malformed_dicom") from None
    except Exception:
        raise IngestError("unreadable_dicom") from None
    return _DicomMetadata(
        study_uid=str(ds.get("StudyInstanceUID", "")),
        series_uid=str(ds.get("SeriesInstanceUID", "")),
        modality=str(ds.get("Modality", "UNKNOWN")),
        sop_class_uid=str(ds.get("SOPClassUID", "")),
        is_multiframe="NumberOfFrames" in ds,
    )


def validate_dicom_file(path: Path) -> None:
    """Validate one DICOM file before metadata extraction.

    Input: A path to a candidate DICOM file.
    Output: None when the file is readable and has valid series identifiers;
        raises ``IngestError`` otherwise. Geometry validation remains the
        converter's responsibility.
    """
    if path.is_symlink():
        raise IngestError("symlink_in_dicom_input")
    if path.is_dir():
        return
    if not path.is_file():
        raise IngestError("invalid_dicom_file")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = pydicom.dcmread(path, stop_before_pixels=True)
    except pydicom.errors.InvalidDicomError:
        raise IngestError("malformed_dicom") from None
    except Exception:
        raise IngestError("unreadable_dicom") from None
    if ds.file_meta.get("MediaStorageSOPClassUID") == MediaStorageDirectoryStorage:
        return
    study_uid = str(ds.get("StudyInstanceUID", ""))
    series_uid = str(ds.get("SeriesInstanceUID", ""))
    if not study_uid or not series_uid:
        raise IngestError("missing_or_invalid_series_identifiers")
    if not UID(study_uid).is_valid or not UID(series_uid).is_valid:
        raise IngestError("missing_or_invalid_series_identifiers")


def _unsupported_reasons(metadata: _DicomMetadata) -> set[str]:
    """Classify supported-series restrictions after validation/extraction.

    Input: Extracted DICOM metadata.
    Output: Safe reason codes explaining why a series cannot be selected.
    """

    modality = metadata.modality
    # Expose only known modality codes, never untrusted free-text tag values.
    modality = modality if modality in ("CT", "MR", "US", "PT", "CR", "DX", "OT") else "UNKNOWN"
    unsupportedReasons = set()
    if modality != "CT":
        unsupportedReasons.add("non_ct_dicom")
    elif metadata.sop_class_uid != str(CTImageStorage) or metadata.is_multiframe:
        unsupportedReasons.add("unsupported_ct_storage_or_multiframe")
    return unsupportedReasons


def _discover(directory: Path) -> list[_Series]:
    """Group valid DICOM files into internal series records.

    Input: A directory containing recursively discoverable DICOM files.
    Output: Internal series records with public candidates and source files.
    A file pydicom does not recognize as DICOM at all (e.g. a filesystem
    artifact like .DS_Store, a stray readme) is not fatal: it is skipped and
    named in a logged warning, so real input is not rejected outright for
    carrying one unrelated file, while the caller can still see what was
    ignored. Other validation failures (symlinks, unreadable/corrupt files
    that were meant to be DICOM, ambiguous series identifiers) remain fatal.
    """
    groups = {}
    ignored = []
    for path in sorted(directory.rglob("*")):
        try:
            validate_dicom_file(path)
            metadata = extract_dicom_meta_data(path)
        except IngestError as error:
            if error.reason != "malformed_dicom":
                raise
            ignored.append(str(path.relative_to(directory)))
            continue
        if metadata is None:
            # Only a directory or recognized DICOMDIR can produce None; other
            # files raise from extract_dicom_meta_data rather than being skipped.
            continue
        study, series = metadata.study_uid, metadata.series_uid
        modality = metadata.modality if metadata.modality in ("CT", "MR", "US", "PT", "CR", "DX", "OT") else "UNKNOWN"
        unsupportedReasons = _unsupported_reasons(metadata)
        files, modalities, rejections = groups.setdefault((study, series), ([], set(), set()))
        files.append(path)
        modalities.add(modality)
        rejections.update(unsupportedReasons)
    result = []
    seen_ids = set()
    for (study, series), (files, modalities, unsupportedReasons) in sorted(groups.items()):
        identifier = _series_id(study, series)
        if identifier in seen_ids:
            raise IngestError("series_identifier_collision")
        seen_ids.add(identifier)
        if len(modalities) != 1:
            unsupportedReasons.add("mixed_modalities")
        modality = "CT" if "CT" in modalities else sorted(modalities)[0]
        candidate = SeriesCandidate(
            identifier, modality, len(files),
            "unsupported" if unsupportedReasons else "unverified", tuple(sorted(unsupportedReasons)),
        )
        result.append(_Series(candidate, files))
    if ignored:
        logger.warning("Non-DICOM files found and ignored: %s", ", ".join(sorted(ignored)))
    return sorted(result, key=lambda item: item.candidate.series_id)


def _dicom_directory(source: Path, workspace: Path) -> Path:
    """Resolve a DICOM directory or safely extract a DICOM zip.

    Input:
        source: DICOM directory or zip path.
        workspace: Caller-owned temporary workspace for zip extraction.
    Output: A directory containing the DICOM files to discover.
    """
    if source.is_dir():
        return source
    if source.is_file() and source.suffix.lower() == ".zip":
        directory = workspace / "dicom"
        try:
            extract_scan_zip(source, directory)
        except Exception:
            raise IngestError("invalid_or_unsafe_zip") from None
        return directory
    raise IngestError("expected_dicom_directory_or_zip")


def list_series(input_path: str | Path) -> tuple[SeriesCandidate, ...]:
    """List candidates without converting; archive contents are removed on return.

    Input: A DICOM directory or zip path.
    Output: Public series candidates with safe IDs, modality, counts, and
        unsupported reasons.
    """
    try:
        source = Path(input_path).expanduser().resolve()
        with TemporaryDirectory(prefix="core-model-discovery-") as workspace:
            return tuple(s.candidate for s in _discover(_dicom_directory(source, Path(workspace))))
    except GenerationError:
        raise
    except Exception:
        raise IngestError("unreadable_input") from None


def _select(series: list[_Series], series_id: str | None) -> _Series:
    """Select exactly one supported series without ranking candidates.

    Input:
        series: Discovered internal series records.
        series_id: Explicit candidate ID, or ``None`` for single-series mode.
    Output: The selected internal series record.
    """
    candidates = tuple(s.candidate for s in series)
    if series_id is not None:
        chosen = next((s for s in series if s.candidate.series_id == series_id), None)
        if chosen is None:
            raise IngestError("unknown_series_id", candidates)
    else:
        ct = [s for s in series if s.candidate.modality == "CT"]
        if not ct:
            raise IngestError("no_ct_series", candidates)
        if len(ct) > 1:
            raise IngestError("series_selection_required", candidates)
        chosen = ct[0]
    if chosen.candidate.support_status == "unsupported":
        raise IngestError("unsupported_series", (chosen.candidate,))
    return chosen


def _convert(directory: Path, destination: Path) -> None:
    """Convert one selected DICOM series with converter validation enabled.

    Input:
        directory: Directory containing only the selected series.
        destination: Output NIfTI path inside the caller workspace.
    Output: None; writes the validated conversion to ``destination``.
    """
    # Do not silently inherit permissive converter settings changed by a caller.
    # Leave global state untouched; the caller can restore the library defaults.
    if not all(getattr(settings, name) for name in REQUIRED_CONVERTER_VALIDATIONS):
        raise IngestError("converter_validation_disabled")
    if settings.resample or settings.pydicom_read_force:
        raise IngestError("unsupported_converter_settings")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dicom2nifti.dicom_series_to_nifti(str(directory), str(destination), reorient_nifti=True)
    except (ConversionError, ConversionValidationError) as error:
        reason = CONVERSION_REASONS.get(str(error), "dicom_conversion_failed")
        raise IngestError(reason) from None
    except Exception:
        raise IngestError("dicom_conversion_failed") from None


def ingest(input_path: str | Path, workspace: str | Path, *, series_id: str | None = None) -> IngestedVolume:
    """Ingest one volume. Workspace must exist and be separate from DICOM input.

    Input:
        input_path: DICOM directory/zip or a valid NIfTI file.
        workspace: Existing caller-owned directory for temporary and converted files.
        series_id: Optional candidate ID required when multiple CT series exist.
    Output: An ``IngestedVolume`` containing the NIfTI path, millimeter geometry,
        and selected candidate (or ``None`` for NIfTI passthrough).
    Successful converted/normalized files live until the caller removes its
    workspace. This function removes its own intermediates on failure. A valid
    mm NIfTI is returned directly, with no input mutation or copy.
    """
    owned = None
    try:
        source = Path(input_path).expanduser().resolve()
        parent = Path(workspace).expanduser().resolve()
        if not source.exists():
            raise IngestError("input_not_found")
        if not parent.is_dir():
            raise IngestError("workspace_not_directory")
        if source.is_dir() and (parent == source or source in parent.parents):
            raise IngestError("workspace_inside_input")
        owned = Path(mkdtemp(prefix="ingest-", dir=parent))
        if source.is_file() and source.name.lower().endswith((".nii", ".nii.gz")):
            if series_id is not None:
                raise IngestError("series_selection_requires_dicom")
            path, geometry = normalize_nifti(source, owned / "volume.nii.gz")
            if path == source:
                owned.rmdir()
            return IngestedVolume(path, geometry)
        series = _discover(_dicom_directory(source, owned))
        chosen = _select(series, series_id)
        selected_dir = owned / "selected"
        selected_dir.mkdir()
        for index, path in enumerate(chosen.files):
            shutil.copyfile(path, selected_dir / f"{index:06d}.dcm")
        destination = owned / "volume.nii.gz"
        _convert(selected_dir, destination)
        path, geometry = normalize_nifti(destination, owned / "normalized.nii.gz")
        shutil.rmtree(selected_dir)
        if (owned / "dicom").exists():
            shutil.rmtree(owned / "dicom")
        return IngestedVolume(path, geometry, chosen.candidate)
    except BaseException as error:
        if owned is not None and owned.exists():
            shutil.rmtree(owned)
        if isinstance(error, (GenerationError, KeyboardInterrupt, SystemExit)):
            raise
        raise IngestError("ingest_failed") from None
