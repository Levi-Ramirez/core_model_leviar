"""Synthetic inputs exercise real DICOM conversion and physical geometry."""

import logging
from pathlib import Path
import stat
from zipfile import ZipFile, ZipInfo

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
import pytest

from core_model.constants.ingest_constants import REQUIRED_CONVERTER_VALIDATIONS
from core_model.contracts import GenerationError
from core_model.ingest import (
    IngestError,
    extract_dicom_meta_data,
    ingest,
    list_series,
    validate_dicom_file,
)
from core_model import ingest as ingest_module
from core_model.utils.nifti import normalize_nifti


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / 'workspace'
    path.mkdir()
    return path


def write_series(directory, *, study=None, series=None, orientation=(1, 0, 0, 0, 1, 0),
                 modality='CT', multiframe=False, count=6):
    directory.mkdir(parents=True, exist_ok=True)
    study, series = study or generate_uid(), series or generate_uid()
    normal = np.cross(orientation[:3], orientation[3:])
    files = []
    for index in range(count):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
        ds.Modality, ds.PatientName, ds.PatientID = modality, 'Synthetic^Patient', 'synthetic-only'
        ds.SeriesDescription = 'Synthetic series description'
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        ds.ImageOrientationPatient = list(orientation)
        ds.ImagePositionPatient = list(np.array([10, 20, 30]) + normal * 4 * index)
        ds.InstanceNumber = index + 1
        ds.PixelSpacing, ds.SliceThickness = [2, 3], 4
        ds.Rows, ds.Columns = 5, 6
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 2, -1024
        if multiframe:
            ds.NumberOfFrames = 1
        ds.PixelData = (index * 100 + np.arange(5)[:, None] * 10 + np.arange(6)).astype('<i2').tobytes()
        path = directory / f'{count - index:03d}.dcm'  # File order is not slice order.
        ds.save_as(path, enforce_file_format=True)
        files.append(path)
    return files


def make_nifti(path, *, units='mm', affine=None, data=None):
    image = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32) if data is None else data,
                           np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units(units)
    nib.save(image, path)
    return image


def zip_directory(directory, target):
    with ZipFile(target, 'w') as archive:
        for path in directory.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(directory))


@pytest.mark.parametrize('orientation', [(1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, 1)])
def test_real_conversion_preserves_world_landmark_and_hu(tmp_path, workspace, orientation):
    directory = tmp_path / 'scan'
    files = write_series(directory, orientation=orientation)
    original = {p: p.read_bytes() for p in files}
    result = ingest(directory, workspace)
    assert result.selected_series == list_series(directory)[0]
    image = nib.load(result.nifti_path)
    # Known pixel: slice 2, row 3, column 4. DICOM coordinates are LPS.
    lps = (np.array([10, 20, 30]) + np.array(orientation[:3]) * 3 * 4
           + np.array(orientation[3:]) * 2 * 3
           + np.cross(orientation[:3], orientation[3:]) * 4 * 2)
    ras = lps * [-1, -1, 1]
    voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), ras)
    np.testing.assert_allclose(voxel, np.round(voxel), atol=1e-6)
    assert image.get_fdata()[tuple(np.round(voxel).astype(int))] == (200 + 30 + 4) * 2 - 1024
    np.testing.assert_allclose(result.geometry.affine, image.affine)
    assert result.geometry.units == 'mm'
    assert all(p.read_bytes() == original[p] for p in files)
    assert not list(workspace.rglob('*.dcm'))


def test_validation_is_a_no_return_gate_before_metadata_extraction(tmp_path):
    files = write_series(tmp_path / 'scan')
    assert validate_dicom_file(files[0]) is None
    metadata = extract_dicom_meta_data(files[0])
    assert metadata.study_uid and metadata.series_uid


def test_discovery_validates_before_extracting_metadata(tmp_path, monkeypatch):
    files = write_series(tmp_path / 'scan', count=1)
    events = []
    original_validate = ingest_module.validate_dicom_file
    original_extract = ingest_module.extract_dicom_meta_data

    def record_validation(path):
        events.append('validate')
        return original_validate(path)

    def record_extraction(path):
        events.append('extract')
        return original_extract(path)

    monkeypatch.setattr(ingest_module, 'validate_dicom_file', record_validation)
    monkeypatch.setattr(ingest_module, 'extract_dicom_meta_data', record_extraction)
    ingest_module._discover(tmp_path / 'scan')
    assert events[:2] == ['validate', 'extract']


def test_multiple_series_require_selection_regardless_of_orientation(tmp_path, workspace, monkeypatch):
    directory = tmp_path / 'scan'
    write_series(directory / 'axial')
    write_series(directory / 'coronal', orientation=(1, 0, 0, 0, 0, 1))
    monkeypatch.setattr(ingest_module, '_convert', lambda *args: pytest.fail('Must not convert before selection'))
    with pytest.raises(IngestError) as caught:
        ingest(directory, workspace)
    assert caught.value.reason == 'series_selection_required'
    assert len(caught.value.candidates) == 2
    assert len(caught.value.selection_commands) == 2
    assert all('--series series_' in command for command in caught.value.selection_commands)
    assert not list(workspace.iterdir())


def test_explicit_selection_converts_only_chosen_series(tmp_path, workspace):
    directory = tmp_path / 'scan'
    write_series(directory / 'one')
    write_series(directory / 'two')
    chosen = list_series(directory)[1]
    result = ingest(directory, workspace, series_id=chosen.series_id)
    assert result.selected_series == chosen


def test_zip_discovery_matches_directory_and_conversion(tmp_path, workspace):
    directory = tmp_path / 'scan'
    write_series(directory / 'nested')
    archive = tmp_path / 'scan.zip'
    zip_directory(directory, archive)
    assert list_series(archive) == list_series(directory)
    result = ingest(archive, workspace)
    assert result.nifti_path.exists()
    assert not list(workspace.rglob('*.dcm'))


def test_unrecognized_files_are_ignored_and_named_in_a_warning(tmp_path, workspace, caplog):
    # A macOS-created zip routinely carries .DS_Store alongside real DICOM;
    # one unrelated file should not reject an otherwise-valid scan outright.
    directory = tmp_path / 'scan'
    write_series(directory)
    (directory / 'readme.txt').write_text('Unrelated export documentation')
    (directory / '.DS_Store').write_bytes(b'not dicom either')
    with caplog.at_level(logging.WARNING, logger='core_model.ingest'):
        result = ingest(directory, workspace)
    assert result.nifti_path.exists()
    assert 'Non-DICOM files found and ignored' in caplog.text
    assert 'readme.txt' in caplog.text
    assert '.DS_Store' in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='core_model.ingest'):
        candidates = list_series(directory)
    assert len(candidates) == 1
    assert 'readme.txt' in caplog.text


def test_symlink_and_unreadable_dicom_remain_fatal(tmp_path, workspace):
    directory = tmp_path / 'scan'
    files = write_series(directory)
    (directory / 'link.dcm').symlink_to(files[0])
    with pytest.raises(IngestError, match='symlink_in_dicom_input'):
        ingest(directory, workspace)


def test_candidate_output_excludes_identifying_metadata(tmp_path):
    files = write_series(tmp_path / 'scan')
    ds = pydicom.dcmread(files[0])
    text = repr(list_series(tmp_path / 'scan'))
    for value in ('Synthetic', 'synthetic-only', str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID), '.dcm'):
        assert value not in text


def test_same_series_uid_in_different_studies_is_not_combined(tmp_path):
    series = generate_uid()
    write_series(tmp_path / 'scan' / 'one', series=series)
    write_series(tmp_path / 'scan' / 'two', series=series)
    assert len(list_series(tmp_path / 'scan')) == 2


def test_id_collision_is_reported(tmp_path, monkeypatch):
    write_series(tmp_path / 'scan' / 'one')
    write_series(tmp_path / 'scan' / 'two')
    monkeypatch.setattr(ingest_module, '_series_id', lambda *args: 'series_collision')
    with pytest.raises(IngestError, match='series_identifier_collision'):
        list_series(tmp_path / 'scan')


@pytest.mark.parametrize('modality,multiframe,reason', [
    ('MR', False, 'non_ct_dicom'), ('CT', True, 'unsupported_ct_storage_or_multiframe'),
])
def test_unsupported_candidates_cannot_be_selected(tmp_path, workspace, modality, multiframe, reason):
    write_series(tmp_path / 'scan', modality=modality, multiframe=multiframe)
    candidate = list_series(tmp_path / 'scan')[0]
    assert reason in candidate.rejection_reasons
    with pytest.raises(IngestError, match='unsupported_series'):
        ingest(tmp_path / 'scan', workspace, series_id=candidate.series_id)


def test_multiple_ct_series_with_one_unsupported_still_require_selection(tmp_path, workspace):
    write_series(tmp_path / 'scan' / 'one')
    write_series(tmp_path / 'scan' / 'two', multiframe=True)
    with pytest.raises(IngestError, match='series_selection_required'):
        ingest(tmp_path / 'scan', workspace)


def test_no_ct_and_unknown_selection(tmp_path, workspace):
    write_series(tmp_path / 'scan', modality='MR')
    with pytest.raises(IngestError, match='no_ct_series'):
        ingest(tmp_path / 'scan', workspace)
    with pytest.raises(IngestError, match='unknown_series_id'):
        ingest(tmp_path / 'scan', workspace, series_id='series_unknown')


@pytest.mark.parametrize('mutation,reason', [('spacing', 'inconsistent_slice_spacing'), ('scout', 'dicom_conversion_failed')])
def test_real_converter_validation_failures(tmp_path, workspace, mutation, reason):
    files = write_series(tmp_path / 'scan')
    targets = files if mutation == 'scout' else files[-1:]
    for path in targets:
        ds = pydicom.dcmread(path)
        if mutation == 'scout':
            ds.ImageType = ['ORIGINAL', 'PRIMARY', 'LOCALIZER']
        else:
            ds.ImagePositionPatient[2] += 5
        ds.save_as(path, enforce_file_format=True)
    with pytest.raises(IngestError, match=reason):
        ingest(tmp_path / 'scan', workspace)
    assert not list(workspace.iterdir())


@pytest.mark.parametrize('setting', REQUIRED_CONVERTER_VALIDATIONS)
def test_disabled_converter_validation_is_not_silently_used(tmp_path, workspace, monkeypatch, setting):
    write_series(tmp_path / 'scan')
    monkeypatch.setattr(ingest_module.settings, setting, False)
    with pytest.raises(IngestError, match='converter_validation_disabled'):
        ingest(tmp_path / 'scan', workspace)
    assert getattr(ingest_module.settings, setting) is False


def test_conversion_failure_does_not_retry_or_expose_raw_exception(tmp_path, workspace, monkeypatch):
    write_series(tmp_path / 'scan' / 'one')
    write_series(tmp_path / 'scan' / 'two')
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise RuntimeError('Synthetic Patient /private/scan.dcm')
    monkeypatch.setattr(ingest_module.dicom2nifti, 'dicom_series_to_nifti', fail)
    with pytest.raises(IngestError) as caught:
        ingest(tmp_path / 'scan', workspace, series_id=list_series(tmp_path / 'scan')[0].series_id)
    assert len(calls) == 1
    assert str(caught.value) == 'ingest: dicom_conversion_failed'
    assert not list(workspace.iterdir())


@pytest.mark.parametrize('member', ['../escape.dcm', '/absolute.dcm', r'C:\scan.dcm', r'..\escape.dcm'])
def test_unsafe_zip_paths_are_rejected(tmp_path, workspace, member):
    archive = tmp_path / 'scan.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr(member, 'synthetic')
    with pytest.raises(IngestError, match='invalid_or_unsafe_zip'):
        ingest(archive, workspace)
    assert not list(workspace.iterdir())
    assert not (tmp_path / 'escape.dcm').exists()


def test_zip_symlink_is_rejected(tmp_path, workspace):
    archive = tmp_path / 'scan.zip'
    with ZipFile(archive, 'w') as z:
        info = ZipInfo('link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, '/synthetic/target')
    with pytest.raises(IngestError, match='invalid_or_unsafe_zip'):
        ingest(archive, workspace)


def test_input_with_no_recognizable_dicom_reports_no_ct_series(tmp_path, workspace, caplog):
    # Every file is ignored individually as non-DICOM; with nothing left to
    # group into a series, this is "nothing found," not a malformed-file error.
    directory = tmp_path / 'scan'
    directory.mkdir()
    (directory / 'bad.dcm').write_text('not dicom')
    with caplog.at_level(logging.WARNING, logger='core_model.ingest'):
        with pytest.raises(IngestError, match='no_ct_series'):
            ingest(directory, workspace)
    assert 'bad.dcm' in caplog.text


@pytest.mark.parametrize('suffix', ['.nii', '.nii.gz'])
def test_mm_nifti_passthrough_is_unchanged(tmp_path, workspace, suffix):
    source = tmp_path / ('input' + suffix)
    make_nifti(source)
    original = source.read_bytes()
    result = ingest(source, workspace)
    assert result.nifti_path == source
    assert result.selected_series is None
    assert source.read_bytes() == original
    assert not list(workspace.iterdir())


@pytest.mark.parametrize('units,scale', [('meter', 1000), ('micron', 0.001)])
def test_nifti_units_normalize_affine_and_preserve_intensity(tmp_path, workspace, units, scale):
    source = tmp_path / 'input.nii.gz'
    affine = np.diag([2, 3, 4, 1.0])
    affine[:3, 3] = [10, 20, 30]
    make_nifti(source, units=units, affine=affine)
    result = ingest(source, workspace)
    expected = affine.copy()
    expected[:3, :] *= scale
    image = nib.load(result.nifti_path)
    np.testing.assert_allclose(image.affine, expected)
    np.testing.assert_allclose(image.get_fdata(), 1)
    assert image.header.get_xyzt_units()[0] == 'mm'
    assert nib.load(source).header.get_xyzt_units()[0] == units


@pytest.mark.parametrize('data,reason', [
    (np.ones((4, 5), dtype=np.float32), 'invalid_volume_dimensions'),
    (np.ones((4, 5, 6, 1), dtype=np.float32), 'invalid_volume_dimensions'),
    (np.ones((4, 5, 1), dtype=np.float32), 'invalid_volume_dimensions'),
    (np.full((4, 5, 6), np.nan), 'nonfinite_volume'),
    (np.full((4, 5, 6), np.inf), 'nonfinite_volume'),
    (np.ones((4, 5, 6), dtype=np.complex64), 'non_scalar_volume'),
])
def test_invalid_nifti_data(tmp_path, workspace, data, reason):
    source = tmp_path / 'input.nii'
    make_nifti(source, data=data)
    with pytest.raises(GenerationError, match=reason):
        ingest(source, workspace)
    assert not list(workspace.iterdir())


@pytest.mark.parametrize('problem,reason', [
    ('unknown_units', 'unknown_spatial_units'), ('no_forms', 'missing_coded_affine'),
    ('singular', 'singular_affine'), ('nonfinite', 'invalid_affine'),
    ('conflict', 'conflicting_affines'),
])
def test_invalid_nifti_geometry(tmp_path, workspace, problem, reason):
    source = tmp_path / 'input.nii'
    image = make_nifti(source)
    header = image.header.copy()
    if problem == 'unknown_units':
        header.set_xyzt_units('unknown')
    elif problem == 'no_forms':
        header['sform_code'], header['qform_code'] = 0, 0
    elif problem == 'singular':
        header['srow_x'] = [0, 0, 0, 0]
    elif problem == 'nonfinite':
        header['srow_x'] = [np.nan, 0, 0, 0]
    else:
        header.set_qform(np.eye(4), code=1)
        header['srow_x'] = [1, 0, 0, 1]
    nib.save(nib.Nifti1Image(image.get_fdata(), None, header), source)
    with pytest.raises(GenerationError, match=reason):
        ingest(source, workspace)


def test_workspace_must_not_be_inside_scan(tmp_path):
    directory = tmp_path / 'scan'
    write_series(directory)
    with pytest.raises(IngestError, match='workspace_inside_input'):
        ingest(directory, directory)


def test_identical_duplicate_slices_follow_converter_behavior(tmp_path, workspace):
    files = write_series(tmp_path / 'scan')
    (tmp_path / 'scan' / 'duplicate.dcm').write_bytes(files[0].read_bytes())
    result = ingest(tmp_path / 'scan', workspace)
    # The converter deduplicates identical slices; discovery still counts files.
    assert result.selected_series.slice_count == 7
    assert nib.load(result.nifti_path).shape == (6, 5, 6)


def test_dicomdir_index_is_ignored(tmp_path):
    directory = tmp_path / 'scan'
    files = write_series(directory)
    from pydicom.uid import MediaStorageDirectoryStorage
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
    ds.file_meta.MediaStorageSOPClassUID = MediaStorageDirectoryStorage
    ds.SOPClassUID = MediaStorageDirectoryStorage
    del ds.StudyInstanceUID, ds.SeriesInstanceUID
    ds.save_as(directory / 'DICOMDIR', enforce_file_format=True)
    assert list_series(directory)[0].slice_count == 6


def test_zip_discovery_with_no_dicom_still_cleans_temporary_extraction(tmp_path, monkeypatch, caplog):
    import tempfile
    archive = tmp_path / 'scan.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('bad.dcm', 'malformed')
    extraction_root = tmp_path / 'extraction'
    extraction_root.mkdir()
    original = tempfile.TemporaryDirectory
    monkeypatch.setattr(ingest_module, 'TemporaryDirectory',
                        lambda **kwargs: original(dir=extraction_root, **kwargs))
    with caplog.at_level(logging.WARNING, logger='core_model.ingest'):
        candidates = list_series(archive)
    assert candidates == ()
    assert 'bad.dcm' in caplog.text
    assert not list(extraction_root.iterdir())


def test_interruption_cleans_owned_workspace_and_preserves_caller_files(tmp_path, workspace, monkeypatch):
    write_series(tmp_path / 'scan')
    sentinel = workspace / 'caller.txt'
    sentinel.write_text('keep')
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(ingest_module, '_convert', interrupt)
    with pytest.raises(KeyboardInterrupt):
        ingest(tmp_path / 'scan', workspace)
    assert list(workspace.iterdir()) == [sentinel]


def test_mm_qform_only_and_close_forms_are_accepted(tmp_path, workspace):
    source = tmp_path / 'input.nii'
    image = make_nifti(source)
    image.set_sform(None, code=0)
    image.set_qform(np.eye(4), code=1)
    nib.save(image, source)
    np.testing.assert_allclose(ingest(source, workspace).geometry.affine, np.eye(4))
    sform = np.eye(4)
    sform[0, 3] = 0.005
    image.set_sform(sform, code=1)
    nib.save(image, source)
    np.testing.assert_allclose(ingest(source, workspace).geometry.affine, sform)


def test_form_tolerance_compares_world_corners_in_mm(tmp_path, workspace):
    source = tmp_path / 'input.nii'
    image = make_nifti(source, units='meter')
    image.set_qform(np.eye(4), code=1)
    sform = np.eye(4)
    sform[0, 0] += 0.00001  # 3 voxels * 0.01 mm exceeds the 0.01 mm tolerance.
    image.set_sform(sform, code=1)
    nib.save(image, source)
    with pytest.raises(GenerationError, match='conflicting_affines'):
        ingest(source, workspace)


def test_unit_normalization_preserves_nifti_slope_intercept(tmp_path, workspace):
    source = tmp_path / 'input.nii'
    image = make_nifti(source, units='meter', data=np.ones((4, 5, 6), dtype=np.int16))
    image.header.set_slope_inter(2, -1024)
    nib.save(image, source)
    result = ingest(source, workspace)
    np.testing.assert_allclose(nib.load(result.nifti_path).get_fdata(), -1022)


@pytest.mark.parametrize('case,reason', [
    ('missing', 'input_not_found'), ('corrupt_zip', 'invalid_or_unsafe_zip'),
    ('corrupt_nifti', 'invalid_or_unreadable_nifti'),
])
def test_unreadable_inputs_report_safe_errors(tmp_path, workspace, case, reason):
    source = tmp_path / {'missing': 'missing', 'corrupt_zip': 'scan.zip', 'corrupt_nifti': 'scan.nii'}[case]
    if case != 'missing':
        source.write_text('Synthetic invalid input')
    with pytest.raises(GenerationError, match=reason):
        ingest(source, workspace)
    assert not list(workspace.iterdir())


def test_explicit_series_id_is_rejected_for_nifti(tmp_path, workspace):
    source = tmp_path / 'input.nii'
    make_nifti(source)
    with pytest.raises(IngestError, match='series_selection_requires_dicom'):
        ingest(source, workspace, series_id='series_123')
