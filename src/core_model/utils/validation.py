"""Validation helpers for choices, identifiers, and portable artifact paths."""

from pathlib import PurePosixPath, PureWindowsPath
import re


def validate_choice(value, choices, name):
    if value not in choices:
        raise ValueError(f"Invalid {name}; expected one of {', '.join(choices)}")


def validate_identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", value):
        raise ValueError("Expected a nonempty class, series, or reason identifier")


def relative_artifact_path(value: str) -> str:
    """Use portable output-relative paths; reject traversal on either platform."""
    value = str(value)
    if (
        not value or value == "." or "\\" in value or "\x00" in value
        or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
        or ".." in PurePosixPath(value).parts
    ):
        raise ValueError("Artifact paths must be relative to the output directory")
    return PurePosixPath(value).as_posix()
