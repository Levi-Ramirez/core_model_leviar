from contextlib import nullcontext

import nibabel as nib
import numpy as np
import pytest

import core_model.segmentation as segmentation


def _write_input(path):
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), np.eye(4)), path)


def test_segment_passes_explicit_full_task_settings_and_validates_masks(tmp_path, monkeypatch):
    input_path = tmp_path / "input.nii.gz"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_input(input_path)
    calls = []

    def fake_api(input_path, **kwargs):
        calls.append((input_path, kwargs))
        for name in ("liver", "spleen"):
            nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.uint8), np.eye(4)),
                     output_dir / f"{name}.nii.gz")

    monkeypatch.setattr(segmentation, "totalsegmentator", fake_api)
    monkeypatch.setattr(segmentation, "_telemetry_disabled", lambda: nullcontext())
    result = segmentation.segment(input_path, output_dir, ["liver", "spleen"], device="cpu", quality="fast")

    assert result.masks == {name: output_dir / f"{name}.nii.gz" for name in ("liver", "spleen")}
    assert calls[0][1] == {
        "output": output_dir, "task": "total", "fast": True, "device": "cpu",
        "roi_subset": None, "preview": False, "statistics": False,
        "radiomics": False, "output_type": "nifti", "quiet": True, "verbose": False,
    }


def test_segment_rejects_unknown_structure_before_inference(tmp_path, monkeypatch):
    input_path = tmp_path / "input.nii.gz"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_input(input_path)
    monkeypatch.setattr(segmentation, "totalsegmentator", lambda: pytest.fail("inference ran"))
    with pytest.raises(segmentation.SegmentationError, match="unknown_structure"):
        segmentation.segment(input_path, output_dir, ["not_a_structure"])


def test_segment_reports_missing_requested_mask(tmp_path, monkeypatch):
    input_path = tmp_path / "input.nii.gz"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    _write_input(input_path)
    monkeypatch.setattr(segmentation, "totalsegmentator", lambda *a, **k: None)
    monkeypatch.setattr(segmentation, "_telemetry_disabled", lambda: nullcontext())
    with pytest.raises(segmentation.SegmentationError, match="missing_mask"):
        segmentation.segment(input_path, output_dir, ["liver"])


def test_telemetry_is_disabled_during_inference_and_restored(monkeypatch):
    state = {"send_usage_stats": True}
    monkeypatch.setattr(segmentation, "setup_totalseg", lambda: None)
    monkeypatch.setattr(segmentation, "get_config_key", lambda key: state.get(key))
    monkeypatch.setattr(segmentation, "set_config_key", lambda key, value: state.__setitem__(key, value))
    with segmentation._telemetry_disabled():
        assert state["send_usage_stats"] is False
    assert state["send_usage_stats"] is True
