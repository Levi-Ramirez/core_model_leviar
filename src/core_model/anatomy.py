"""Resolve anatomy selection and stable colors without importing inference code."""

import colorsys
from dataclasses import dataclass
import hashlib
from types import MappingProxyType

from core_model.constants.anatomy_constants import (
    COLOR_MAP_VERSION,
    CONTEXT_COLOR,
    DEFAULT_STRUCTURE_COLORS,
    DEFAULT_STRUCTURES,
    SPINE_AND_PELVIS_STRUCTURES,
    TOTALSEGMENTATOR_VERSION,
)
from core_model.constants.segmentator_class_map import TOTAL_CLASS_MAP
from core_model.contracts import GenerationOptions

SUPPORTED_STRUCTURES = tuple(TOTAL_CLASS_MAP.values())
_SUPPORTED_NAMES = frozenset(SUPPORTED_STRUCTURES)
_CONTEXT_NAMES = frozenset(SPINE_AND_PELVIS_STRUCTURES)


def _color_for_name(name: str) -> tuple[int, int, int]:
    if name in _CONTEXT_NAMES:
        return CONTEXT_COLOR
    if name in DEFAULT_STRUCTURE_COLORS:
        return DEFAULT_STRUCTURE_COLORS[name]
    # A stable digest avoids randomized Python hashes and selection-order colors.
    # This is a display palette, not an anatomical or diagnostic color convention.
    digest = hashlib.sha256(f"{COLOR_MAP_VERSION}:{name}".encode("ascii")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65536
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.6, 0.85))


STRUCTURE_COLORS = MappingProxyType({name: _color_for_name(name) for name in SUPPORTED_STRUCTURES})


@dataclass(frozen=True)
class AnatomySelection:
    """Interest/context roles are retained even when their names overlap."""

    interests: tuple[str, ...]
    context: tuple[str, ...]

    @property
    def structures(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.interests, *self.context)))


def resolve_selection(options: GenerationOptions) -> AnatomySelection:
    """Resolve names before inference; mask presence is assessed later by model.py.

    Input: Generation options containing the include/all-structures selection.
    Output: An ``AnatomySelection`` with interest and context names, or a
        ``ValueError`` for an unknown class.
    """
    if options.all_structures:
        interests = SUPPORTED_STRUCTURES
    elif options.include is not None:
        interests = options.include
    else:
        interests = DEFAULT_STRUCTURES
    unknown = set(interests) - _SUPPORTED_NAMES
    if unknown:
        raise ValueError(
            f"Unknown total-task classes: {', '.join(sorted(unknown))}. "
            f"Valid names: {', '.join(SUPPORTED_STRUCTURES)}"
        )
    return AnatomySelection(interests=interests, context=SPINE_AND_PELVIS_STRUCTURES)


def structure_color(name: str) -> tuple[int, int, int]:
    """Return the same RGB color for a class in every selection and artifact.

    Input: A supported TotalSegmentator class name.
    Output: An RGB tuple with integer channels from 0 through 255.
    """
    try:
        return STRUCTURE_COLORS[name]
    except KeyError:
        raise ValueError("Unknown total-task class; use a supported structure name") from None
