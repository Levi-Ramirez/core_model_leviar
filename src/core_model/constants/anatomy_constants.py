"""Anatomy selection and display defaults for TotalSegmentator's total task."""

from types import MappingProxyType

TOTALSEGMENTATOR_VERSION = "2.18.0"
COLOR_MAP_VERSION = 1
CONTEXT_COLOR = (200, 200, 200)

DEFAULT_STRUCTURES = (
    "liver", "spleen", "kidney_left", "kidney_right", "gallbladder", "stomach",
    "pancreas", "adrenal_gland_left", "adrenal_gland_right", "duodenum",
    "small_bowel", "colon", "urinary_bladder",
)

# Explicit colors keep the default organ legend distinct. These are display
# choices, not standardized anatomical colors or segmentation confidence.
DEFAULT_STRUCTURE_COLORS = MappingProxyType({
    "liver": (180, 70, 60),
    "spleen": (140, 90, 170),
    "kidney_left": (210, 100, 120),
    "kidney_right": (160, 60, 100),
    "gallbladder": (80, 150, 80),
    "stomach": (220, 150, 80),
    "pancreas": (200, 175, 70),
    "adrenal_gland_left": (100, 160, 190),
    "adrenal_gland_right": (60, 110, 170),
    "duodenum": (170, 130, 80),
    "small_bowel": (100, 175, 150),
    "colon": (70, 140, 140),
    "urinary_bladder": (100, 110, 195),
})

SPINE_AND_PELVIS_STRUCTURES = (
    *(f"vertebrae_C{level}" for level in range(1, 8)),
    *(f"vertebrae_T{level}" for level in range(1, 13)),
    *(f"vertebrae_L{level}" for level in range(1, 6)),
    "sacrum", "hip_left", "hip_right",
)
