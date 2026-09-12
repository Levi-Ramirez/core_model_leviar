"""Validate a scalar CT volume and normalize declared spatial units."""

from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np

from core_model.constants.ingest_constants import FORM_TOLERANCE_MM, UNIT_SCALE_TO_MM
from core_model.contracts import GenerationError, ImageGeometry


def _fail(reason):
    raise GenerationError("ingest", reason)


def _validate_affine(affine):
    if not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]):
        _fail("invalid_affine")
    if np.linalg.slogdet(affine[:3, :3])[0] == 0:
        _fail("singular_affine")


def normalize_nifti(source: Path, destination: Path) -> tuple[Path, ImageGeometry]:
    """Return the source for valid mm input; write a normalized copy otherwise.

    Input:
        source: NIfTI path to validate and, when necessary, convert to mm.
        destination: Output path for a normalized copy.
    Output: ``(path, geometry)`` for a validated millimeter NIfTI and its
        serialized image geometry.
    Values are read after NIfTI intensity scaling. CT/HU provenance remains a
    caller obligation. Source files are never modified or externally fetched.
    """
    try:
        image = nib.load(source)
        if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
            _fail("unsupported_volume_format")
        if len(image.shape) != 3 or any(size < 2 for size in image.shape):
            _fail("invalid_volume_dimensions")
        if image.get_data_dtype().kind not in "iuf":
            _fail("non_scalar_volume")
        unit, _ = image.header.get_xyzt_units()
        if unit not in UNIT_SCALE_TO_MM:
            _fail("unknown_spatial_units")
        scale = UNIT_SCALE_TO_MM[unit]
        forms = []
        for getter in (image.get_sform, image.get_qform):
            affine, code = getter(coded=True)
            if code:
                affine = affine.copy()
                affine[:3, :] *= scale
                _validate_affine(affine)
                forms.append(affine)
        if not forms:
            _fail("missing_coded_affine")
        if len(forms) == 2:
            corners = np.array(list(product(*( (0, n - 1) for n in image.shape))))
            distance = np.linalg.norm(
                nib.affines.apply_affine(forms[0], corners)
                - nib.affines.apply_affine(forms[1], corners), axis=1,
            )
            if np.any(distance > FORM_TOLERANCE_MM):
                _fail("conflicting_affines")
        # Avoid a second full-volume float64 allocation for already-mm inputs.
        for index in range(image.shape[2]):
            if not np.isfinite(np.asanyarray(image.dataobj[:, :, index])).all():
                _fail("nonfinite_volume")
        affine = forms[0]  # Same precedence as nibabel: coded sform, then qform.
        geometry = ImageGeometry(tuple(int(n) for n in image.shape), tuple(map(tuple, affine)))
        if scale == 1:
            return source, geometry
        data = np.asanyarray(image.dataobj)
        # A fresh header omits incidental patient-identifying header text/extensions.
        normalized = type(image)(data, affine, dtype=data.dtype)
        normalized.header.set_xyzt_units("mm")
        normalized.set_sform(affine, code=1)
        normalized.set_qform(None, code=0)  # sform preserves any legitimate shear.
        nib.save(normalized, destination)
        # Report serialized geometry, including NIfTI-1 float32 rounding.
        return normalize_nifti(destination, destination)
    except GenerationError:
        raise
    except Exception:
        raise GenerationError("ingest", "invalid_or_unreadable_nifti") from None
