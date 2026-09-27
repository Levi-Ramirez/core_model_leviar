"""Run TotalSegmentator and validate the masks it writes."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from totalsegmentator.config import get_config_key, set_config_key, setup_totalseg
from totalsegmentator.python_api import totalsegmentator

from core_model.anatomy import SUPPORTED_STRUCTURES
from core_model.contracts import GenerationError
from core_model.constants.contracts_constants import SEGMENTATION_TASK


class SegmentationError(GenerationError):
    """Safe error raised for inference setup, execution, or output failures.

    Input:
        reason: A stable, non-identifying segmentation error code.
    Output:
        An exception carrying the safe reason for the segmentation stage.
    """

    def __init__(self, reason: str):
        super().__init__("segmentation", reason)


@dataclass(frozen=True)
class SegmentationResult:
    """Validated masks produced for the requested structures.

    Input:
        masks: Mapping of supported structure structure_names to NIfTI mask paths.
        output_dir: Directory containing the complete TotalSegmentator output.
        device: Device passed to TotalSegmentator.
        quality: Quality passed to TotalSegmentator.
    Output:
        An immutable record used by later mesh and pipeline stages.
    """

    masks: dict[str, Path]
    output_dir: Path
    device: str
    quality: str


def _normalise_structures(structures: Iterable[str]) -> tuple[str, ...]:
    """Validate and deduplicate requested output structures.

    Input: An iterable of TotalSegmentator class structure_names.
    Output: structure_names in first-seen order, or a ``SegmentationError``.
    """
    structure_names = tuple(dict.fromkeys(structures))
    unknown = sorted(set(structure_names) - set(SUPPORTED_STRUCTURES))
    if unknown:
        raise SegmentationError("unknown_structure")
    if not structure_names:
        raise SegmentationError("no_structures_requested")
    return structure_names


@contextmanager
def _telemetry_disabled():
    """Disable TotalSegmentator usage statistics while inference runs.

    Input: None.
    Output: A context in which ``send_usage_stats`` is false.
    """
    try:
        setup_totalseg()
        previous = get_config_key("send_usage_stats")
        set_config_key("send_usage_stats", False)
    except Exception:
        raise SegmentationError("telemetry_configuration_failed") from None
    try:
        yield
    finally:
        # Restore the caller's setting when one existed. The temporary false
        # value is the only configuration change made by this adapter.
        if previous is not None:
            try:
                set_config_key("send_usage_stats", previous)
            except Exception:
                pass


def _validate_mask(path: Path) -> None:
    """Validate one generated mask's basic NIfTI representation.

    Input: A path to a generated NIfTI mask.
    Output: None when valid; raises ``SegmentationError`` otherwise.
    """
    if not path.is_file():
        raise SegmentationError("missing_mask")
    try:
        image = nib.load(path)
        data = np.asanyarray(image.dataobj)
        affine = np.asarray(image.affine)
    except Exception:
        raise SegmentationError("invalid_mask") from None
    if data.ndim != 3 or not np.issubdtype(data.dtype, np.number):
        raise SegmentationError("invalid_mask")
    if not np.isfinite(data).all() or not np.isfinite(affine).all():
        raise SegmentationError("invalid_mask")
    if affine.shape != (4, 4) or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12:
        raise SegmentationError("invalid_mask")


def segment(input_path: Path, output_dir: Path, structures: Iterable[str], *,
            device: str = "cpu", quality: str = "standard") -> SegmentationResult:
    """Run TotalSegmentator's full ``total`` task and validate requested masks.

    Input:
        input_path: Validated NIfTI volume to segment.
        output_dir: Caller-owned empty directory for TotalSegmentator outputs.
        structures: Requested class structure_names whose masks will be returned.
        device: Explicit ``cpu``, ``gpu``, or ``mps`` device selection.
        quality: ``standard`` or ``fast`` resolution selection.
    Output:
        A ``SegmentationResult`` with validated requested mask paths.
    """
    if device not in {"cpu", "gpu", "mps"}:
        raise SegmentationError("unavailable_device")
    if quality not in {"standard", "fast"}:
        raise SegmentationError("invalid_quality")
    structure_names = _normalise_structures(structures)
    input_path, output_dir = Path(input_path), Path(output_dir)
    if not input_path.is_file() or not output_dir.is_dir():
        raise SegmentationError("invalid_segmentation_paths")
    try:
        with _telemetry_disabled():
            totalsegmentator(
                input_path,
                output=output_dir,
                task=SEGMENTATION_TASK,
                fast=quality == "fast",
                device=device,
                roi_subset=None,
                preview=False,
                statistics=False,
                radiomics=False,
                output_type="nifti",
                # Keep upstream model/download progress visible to CLI users.
                quiet=False,
                verbose=False,
            )
    except SegmentationError:
        raise
    except Exception as error:
        message = str(error).lower()
        reason = "download_failed" if any(word in message for word in ("download", "weight", "internet", "network")) else "inference_failed"
        raise SegmentationError(reason) from None
    masks = {}
    for name in structure_names:
        path = output_dir / f"{name}.nii.gz"
        _validate_mask(path)
        masks[name] = path
    return SegmentationResult(masks=masks, output_dir=output_dir, device=device, quality=quality)


run_segmentation = segment
