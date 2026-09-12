"""Extract scan archives into a new, caller-owned directory."""

from pathlib import Path
import shutil
import stat
from zipfile import ZipFile

from core_model.utils.validation import relative_artifact_path


def extract_scan_zip(source: Path, destination: Path) -> None:
    """Extract a scan archive after rejecting unsafe entries.

    Input:
        source: Zip archive containing DICOM files.
        destination: New directory owned by the caller for extracted files.
    Output: None; writes regular files and directories beneath ``destination``.
    """
    with ZipFile(source) as archive:
        entries = []
        seen = set()
        for info in archive.infolist():
            name = relative_artifact_path(info.filename)
            mode = stat.S_IFMT(info.external_attr >> 16)
            if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError("Archive contains a link or special file")
            if name in seen:
                raise ValueError("Archive contains duplicate destinations")
            seen.add(name)
            entries.append((info, Path(name)))
        destination.mkdir()
        for info, relative in entries:
            target = destination / relative
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
