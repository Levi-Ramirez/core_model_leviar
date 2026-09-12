import json
from pathlib import Path

import pytest

from core_model.contracts import (
    Artifact, GenerationError, GenerationManifest, GenerationOptions,
    GenerationResult, ImageGeometry, SeriesCandidate, StageIssue, StructureOutcome,
)


def test_options_defaults_and_snapshot():
    defaults = GenerationOptions()
    assert (defaults.device, defaults.quality, defaults.include) == ('cpu', 'standard', None)
    names = ['liver', 'spleen', 'liver']
    options = GenerationOptions(include=names)
    names.append('colon')
    assert options.include == ('liver', 'spleen')


@pytest.mark.parametrize('kwargs', [
    {'include': []}, {'include': 'liver'},
    {'include': ['liver'], 'all_structures': True},
    {'device': 'auto'}, {'quality': 'best'}, {'all_structures': 'yes'},
    {'series_id': ''}, {'include': ['../liver']},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        GenerationOptions(**kwargs)


def test_series_discovery_does_not_claim_valid_geometry():
    candidate = SeriesCandidate('series_123', 'CT', 20)
    assert candidate.support_status == 'unverified'
    rejected = SeriesCandidate('series_456', 'CT', 1, 'unsupported', ('multiframe',))
    assert rejected.rejection_reasons == ('multiframe',)
    with pytest.raises(ValueError):
        SeriesCandidate('series_456', 'CT', 1, 'unsupported')


@pytest.mark.parametrize('path', [
    '/tmp/patient/model.ply', '../model.ply', 'structures/../../model.ply',
    'C:/patient/model.ply', 'C:model.ply', r'..\model.ply',
    r'\\server\share\model.ply', '', '.', 'bad\x00.ply',
])
def test_artifact_paths_cannot_escape_output_directory(path):
    with pytest.raises(ValueError):
        Artifact(path, 'combined_mesh')


def test_structure_outcomes_distinguish_skips_failures_and_clipping():
    included = StructureOutcome('liver', 'included', clipped=True, color=(200, 80, 60))
    skipped = StructureOutcome('spleen', 'skipped', reason='empty_mask')
    failed = StructureOutcome('colon', 'failed', reason='missing_mask')
    result = GenerationResult(Path('out'), 'partial', structures=[included, skipped],
                              errors=[StageIssue('render', 'render_failed', 'liver')])
    assert result.included_classes == ('liver',)
    assert result.skipped_classes == {'spleen': 'empty_mask'}
    assert included.clipped
    assert failed.reason == 'missing_mask'
    with pytest.raises(ValueError):
        StructureOutcome('spleen', 'skipped')


def test_manifest_serialization_contains_provenance_without_input_metadata():
    geometry = ImageGeometry((4, 5, 6), ((2, 0, 0, 10), (0, 3, 0, -5),
                                        (0, 0, 4, 1), (0, 0, 0, 1)))
    manifest = GenerationManifest(
        options=GenerationOptions(series_id='series_123', include=['liver']),
        status='complete', resolved_device='cpu',
        software_versions={'core-model-leviar': '0.1.0'}, weight_identifier='model_checksum',
        source_geometry=geometry,
        artifacts=[Artifact('model.ply', 'combined_mesh'),
                   Artifact('structures/liver.ply', 'structure_mesh', 'liver')],
        structures=[StructureOutcome('liver', 'included', color=(200, 80, 60), geometry=geometry)],
        stage_timings={'ingest': 0.2},
    )
    # Arbitrary incidental attributes are not part of the serialized schema.
    manifest.input_path = '/synthetic-patient/private/scan.dcm'
    manifest.patient_name = 'Synthetic Patient'
    manifest.raw_exception = RuntimeError('Synthetic Patient')
    encoded = manifest.to_json()
    data = json.loads(encoded)
    assert data['schema_version'] == 1
    assert data['task'] == 'total'
    assert data['options']['include'] == ['liver']
    assert data['source_geometry']['affine'][0] == [2, 0, 0, 10]
    assert data['structures'][0]['geometry']['units'] == 'mm'
    assert data['artifacts'][1]['path'] == 'structures/liver.ply'
    assert 'Synthetic Patient' not in encoded
    assert 'scan.dcm' not in encoded
    assert 'input_path' not in data
    with pytest.raises(TypeError):
        GenerationManifest(patient_name='Synthetic Patient')


def test_partial_result_serializes_paths_and_preview_errors():
    result = GenerationResult(
        Path('out'), 'partial', artifacts=[Artifact('model.ply', 'combined_mesh')],
        errors=[StageIssue('render', 'render_failed', 'liver')],
    )
    data = json.loads(json.dumps(result.to_dict()))
    assert data['manifest_path'] == str(Path('out/manifest.json'))
    assert data['artifacts'][0]['path'] == 'model.ply'
    assert data['errors'][0]['structure'] == 'liver'


def test_fatal_error_does_not_display_private_paths_or_upstream_exception():
    error = GenerationError('segmentation', 'inference_failed',
                            manifest_path=Path('/synthetic-patient/out/manifest.json'))
    assert str(error) == 'segmentation: inference_failed'
    assert error.stage == 'segmentation'
    assert error.manifest_path.name == 'manifest.json'
    with pytest.raises(ValueError):
        GenerationError('segmentation', 'Failed reading /synthetic-patient/scan.dcm')


@pytest.mark.parametrize('status', ['running', 'complete', 'partial', 'failed'])
def test_manifest_statuses(status):
    errors = [StageIssue('render', 'render_failed')] if status in ('partial', 'failed') else []
    assert json.loads(GenerationManifest(status=status, errors=errors).to_json())['status'] == status


@pytest.mark.parametrize('status,errors', [
    ('complete', [StageIssue('render', 'render_failed')]), ('partial', []), ('failed', []),
])
def test_inconsistent_manifest_status_is_rejected(status, errors):
    with pytest.raises(ValueError):
        GenerationManifest(status=status, errors=errors).to_json()


@pytest.mark.parametrize('seconds', [-1, float('nan'), float('inf')])
def test_invalid_timing_cannot_be_serialized(seconds):
    with pytest.raises(ValueError):
        GenerationManifest(stage_timings={'mesh': seconds}).to_json()


def test_duplicate_artifacts_and_structures_are_rejected():
    mesh = Artifact('model.ply', 'combined_mesh')
    structure = StructureOutcome('liver', 'included')
    with pytest.raises(ValueError):
        GenerationManifest(artifacts=[mesh, mesh]).to_json()
    with pytest.raises(ValueError):
        GenerationManifest(structures=[structure, structure]).to_json()


def test_manifest_collections_are_not_shared_between_runs():
    first, second = GenerationManifest(), GenerationManifest()
    first.artifacts.append(Artifact('model.ply', 'combined_mesh'))
    assert second.artifacts == []


@pytest.mark.parametrize('kwargs', [
    {'errors': [RuntimeError('Synthetic Patient')]},
    {'warnings': [{'message': 'Synthetic Patient'}]},
    {'options': {'patient_name': 'Synthetic Patient'}},
    {'source_geometry': {'patient_name': 'Synthetic Patient'}},
])
def test_raw_metadata_and_exceptions_cannot_replace_manifest_records(kwargs):
    with pytest.raises(ValueError):
        GenerationManifest(**kwargs).to_json()


def test_fatal_failures_cannot_be_reported_as_partial_success():
    with pytest.raises(ValueError):
        GenerationResult(Path('out'), 'partial',
                         errors=[StageIssue('segmentation', 'inference_failed')])
    with pytest.raises(ValueError):
        GenerationManifest(status='complete', structures=[
            StructureOutcome('liver', 'failed', reason='missing_mask')
        ]).to_json()
