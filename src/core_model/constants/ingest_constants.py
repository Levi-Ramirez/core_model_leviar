"""Input-format and physical-geometry constants."""

FORM_TOLERANCE_MM = 0.01
UNIT_SCALE_TO_MM = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
REQUIRED_CONVERTER_VALIDATIONS = (
    "validate_slicecount", "validate_orientation", "validate_orthogonal",
    "validate_slice_increment", "validate_multiframe_implicit",
)
CONVERSION_REASONS = {
    "TOO_FEW_SLICES/LOCALIZER": "too_few_slices_or_localizer",
    "IMAGE_ORIENTATION_INCONSISTENT": "inconsistent_orientation",
    "SLICE_INCREMENT_INCONSISTENT": "inconsistent_slice_spacing",
    "INSTANCE_NUMBER_INCONSISTENT": "inconsistent_instance_numbers",
    "NON_CUBICAL_IMAGE/GANTRY_TILT": "unsupported_gantry_tilt",
    "NOT_A_VOLUME": "not_a_volume",
    "NON_IMAGING_DICOM_FILES": "non_imaging_dicom",
    "NO_DICOM_FILES_FOUND": "no_convertible_slices",
    "MISSING_DICOM_FILES": "missing_dicom_slices",
    "UNSUPPORTED_DATA": "unsupported_dicom_data",
}
