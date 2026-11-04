"""Shared test infrastructure; real scans are always supplied explicitly."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


def pytest_addoption(parser):
    parser.addoption(
        "--scan-path",
        metavar="PATH",
        help="Local DICOM directory/zip or NIfTI for opt-in integration tests",
    )


@pytest.fixture(scope="session")
def opengl(tmp_path_factory):
    """Skip tests that draw only when this machine has no usable renderer."""
    import trimesh

    from core_model.render import render_previews

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), geom_name="liver")
    result = render_previews(scene, tmp_path_factory.mktemp("opengl_probe"))
    if "renderer_unavailable" in result.failures.values():
        pytest.skip("Preview rendering is unavailable on this machine")
    # Anything else is a rendering bug, which must fail rather than skip.
    assert not result.failures, result.failures


@pytest.fixture
def local_scan_path(request):
    """Require a local input without collecting or downloading patient data."""
    if request.node.get_closest_marker("integration") is None:
        pytest.fail("local_scan_path requires @pytest.mark.integration")
    value = request.config.getoption("--scan-path")
    if not value:
        pytest.skip("Supply a local scan with --scan-path PATH to run this integration test")
    scan_path = Path(value).expanduser().resolve()
    if not scan_path.exists():
        pytest.fail("--scan-path must refer to an existing local file or directory")
    return scan_path
