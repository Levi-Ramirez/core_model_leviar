from pathlib import Path

import core_model.cli as cli
from core_model.contracts import GenerationError, GenerationResult, StageIssue


def test_generate_forwards_options_and_returns_success(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_generate(input_path, output_dir, options):
        captured.update(input_path=input_path, output_dir=output_dir, options=options)
        return GenerationResult(Path(output_dir), "complete")

    monkeypatch.setattr(cli, "generate_model", fake_generate)
    code = cli.main(["generate", "scan", "--output", "out", "--series", "series_abc",
                     "--include", "liver", "spleen", "--device", "mps", "--quality", "fast"])
    assert code == 0
    assert captured["options"].include == ("liver", "spleen")
    assert captured["options"].device == "mps"
    assert "status: complete" in capsys.readouterr().out


def test_list_series_prints_safe_candidates(monkeypatch, capsys):
    from core_model.contracts import SeriesCandidate
    monkeypatch.setattr(cli, "list_series", lambda path: (
        SeriesCandidate("series_abc", "CT", 12),
    ))
    assert cli.main(["list-series", "scan"]) == 0
    assert "series_abc\tCT\t12\tunverified" in capsys.readouterr().out


def test_selection_error_prints_rerun_and_returns_two(monkeypatch, capsys):
    error = GenerationError("ingest", "series_selection_required")
    error.candidates = ()
    monkeypatch.setattr(cli, "generate_model", lambda *args: (_ for _ in ()).throw(error))
    assert cli.main(["generate", "scan", "--output", "out"]) == 2
    assert "ingest: series_selection_required" in capsys.readouterr().err


def test_partial_result_returns_three(monkeypatch, tmp_path, capsys):
    result = GenerationResult(
        tmp_path, "partial", errors=[StageIssue("render", "preview_failed")],
    )
    monkeypatch.setattr(cli, "generate_model", lambda *args: result)
    assert cli.main(["generate", "scan", "--output", "out"]) == 3
    assert "status: partial" in capsys.readouterr().out
