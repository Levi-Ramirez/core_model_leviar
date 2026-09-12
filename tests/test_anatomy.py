"""Verify the supported map against the installed release without model imports."""

import ast
from importlib.metadata import distribution
import json
import subprocess
import sys

import pytest

from core_model.anatomy import (
    CONTEXT_COLOR, DEFAULT_STRUCTURES, SPINE_AND_PELVIS_STRUCTURES,
    STRUCTURE_COLORS, SUPPORTED_STRUCTURES, TOTAL_CLASS_MAP,
    TOTALSEGMENTATOR_VERSION, resolve_selection, structure_color,
)
from core_model.contracts import GenerationOptions


def test_map_matches_installed_totalsegmentator_release():
    package = distribution('TotalSegmentator')
    assert package.version == TOTALSEGMENTATOR_VERSION
    path = package.locate_file('totalsegmentator/map_to_binary.py')
    tree = ast.parse(path.read_text())
    # Read upstream's data assignment without executing package code or inference.
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'class_map'
            for target in node.targets
        ):
            upstream = ast.literal_eval(node.value)['total']
            break
    else:
        pytest.fail('Pinned package no longer has a literal class_map; review the adapter')
    assert dict(TOTAL_CLASS_MAP) == upstream
    assert set(DEFAULT_STRUCTURES) <= set(upstream.values())
    assert set(SPINE_AND_PELVIS_STRUCTURES) <= set(upstream.values())


def test_default_selection_includes_organs_and_context_once():
    selection = resolve_selection(GenerationOptions())
    assert selection.interests == DEFAULT_STRUCTURES
    assert selection.context == SPINE_AND_PELVIS_STRUCTURES
    assert len(selection.structures) == len(set(selection.structures))
    assert {'liver', 'colon', 'vertebrae_L5', 'sacrum', 'hip_left', 'hip_right'} <= set(selection.structures)


def test_explicit_context_interest_retains_roles_without_duplicate_meshes():
    selection = resolve_selection(GenerationOptions(include=['liver', 'hip_left', 'liver']))
    assert selection.interests == ('liver', 'hip_left')
    assert 'hip_left' in selection.context
    assert selection.structures.count('hip_left') == 1
    assert 'colon' not in selection.structures


def test_all_selection_uses_every_supported_class_once():
    selection = resolve_selection(GenerationOptions(all_structures=True))
    assert selection.interests == SUPPORTED_STRUCTURES
    assert set(selection.structures) == set(TOTAL_CLASS_MAP.values())
    assert len(selection.structures) == len(TOTAL_CLASS_MAP)


def test_unknown_class_fails_with_valid_name_guidance():
    with pytest.raises(ValueError, match='Unknown total-task classes: imaginary_organ.*Valid names:'):
        resolve_selection(GenerationOptions(include=['imaginary_organ']))
    with pytest.raises(ValueError, match='Unknown total-task class'):
        structure_color('imaginary_organ')


def test_colors_are_selection_independent_and_context_is_subdued():
    for names in (['liver'], ['colon', 'liver'], ['liver', 'hip_left']):
        selection = resolve_selection(GenerationOptions(include=names))
        colors = {name: structure_color(name) for name in selection.structures}
        assert colors['liver'] == STRUCTURE_COLORS['liver']
        assert colors['hip_left'] == CONTEXT_COLOR
    assert set(STRUCTURE_COLORS) == set(SUPPORTED_STRUCTURES)
    assert all(len(color) == 3 and all(0 <= c <= 255 for c in color)
               for color in STRUCTURE_COLORS.values())
    assert len({structure_color(name) for name in DEFAULT_STRUCTURES}) == len(DEFAULT_STRUCTURES)


def test_maps_are_read_only():
    with pytest.raises(TypeError):
        TOTAL_CLASS_MAP[1] = 'changed'
    with pytest.raises(TypeError):
        STRUCTURE_COLORS['liver'] = (0, 0, 0)


def test_anatomy_import_is_lightweight_and_colors_stable_across_processes():
    code = '''
import json
import sys
from core_model.anatomy import structure_color
assert not {'torch', 'nnunetv2', 'totalsegmentator', 'numpy'} & set(sys.modules)
print(json.dumps(structure_color('liver')))
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert tuple(json.loads(result.stdout)) == structure_color('liver')
