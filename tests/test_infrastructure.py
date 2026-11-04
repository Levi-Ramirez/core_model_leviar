"""Exercise the actual pytest configuration in isolated synthetic projects."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured_pytester(pytester):
    # Copy the real settings so regressions in pyproject/conftest are exercised.
    pytester.makefile(
        ".toml", pyproject=(PROJECT_ROOT / "pyproject.toml").read_text()
    )
    pytester.makeconftest((PROJECT_ROOT / "tests/conftest.py").read_text())
    (pytester.path / "tests").mkdir()
    return pytester


def test_default_and_explicit_integration_selection(configured_pytester):
    runner = configured_pytester
    runner.makepyfile(
        test_selection="""
        import pytest

        def test_unit():
            pass

        @pytest.mark.integration
        def test_real_model_placeholder():
            pass
        """
    )
    runner.runpytest_subprocess("test_selection.py").assert_outcomes(
        passed=1, deselected=1
    )
    runner.runpytest_subprocess("test_selection.py", "-m", "integration").assert_outcomes(
        passed=1, deselected=1
    )


def test_unit_network_is_blocked_and_integration_can_enable_it(configured_pytester):
    runner = configured_pytester
    runner.makepyfile(
        test_network="""
        import socket
        import pytest
        from pytest_socket import SocketBlockedError

        def test_unit():
            with pytest.raises(SocketBlockedError):
                socket.socket()

        @pytest.mark.integration
        @pytest.mark.enable_socket
        def test_download_capability():
            # Create and close a socket without making any network request.
            with socket.socket():
                pass
        """
    )
    runner.runpytest_subprocess("test_network.py").assert_outcomes(passed=1, deselected=1)
    runner.runpytest_subprocess("test_network.py", "-m", "integration").assert_outcomes(
        passed=1, deselected=1
    )


def test_integration_scan_is_explicit_and_local(configured_pytester):
    runner = configured_pytester
    runner.makepyfile(
        test_scan="""
        import pytest

        @pytest.mark.integration
        def test_input(local_scan_path):
            assert local_scan_path.is_dir()
        """
    )
    runner.runpytest_subprocess("test_scan.py", "-m", "integration").assert_outcomes(
        skipped=1
    )
    scan_dir = runner.path / "synthetic_scan"
    scan_dir.mkdir()
    runner.runpytest_subprocess(
        "test_scan.py", "-m", "integration", "--scan-path", str(scan_dir)
    ).assert_outcomes(passed=1)
    runner.runpytest_subprocess(
        "test_scan.py", "-m", "integration", "--scan-path", str(scan_dir / "missing")
    ).assert_outcomes(errors=1)


def test_cannot_request_patient_input(configured_pytester):
    runner = configured_pytester
    runner.makepyfile(
        test_scan="""
        def test_input(local_scan_path):
            pass
        """
    )
    runner.runpytest_subprocess("test_scan.py").assert_outcomes(errors=1)


@pytest.mark.parametrize(
    "reason, outcome",
    [("renderer_unavailable", {"skipped": 1}), ("preview_failed", {"errors": 1})],
    ids=["no_renderer_skips", "rendering_bug_fails"],
)
def test_opengl_fixture_only_skips_when_the_renderer_is_unavailable(
    configured_pytester, reason, outcome
):
    configured_pytester.makepyfile(
        test_probe=f"""
        import core_model.render as render

        render.render_previews = lambda *args, **kwargs: render.RenderResult(
            {{}}, {{"combined": "{reason}"}}
        )

        def test_needs_rendering(opengl):
            pass
        """
    )
    configured_pytester.runpytest_subprocess("test_probe.py").assert_outcomes(**outcome)
