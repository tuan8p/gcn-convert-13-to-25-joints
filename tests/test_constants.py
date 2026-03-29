"""Tests for ssr_gcn.constants — no torch required."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssr_gcn.constants import (
    JOINT_NAMES_NTU_25,
    JOINT_NAMES_TOYOTA_13,
    MISSING_NTU_JOINTS,
    NTU_EDGES,
    TOYOTA_TO_NTU_25_MAP,
    VISIBLE_NTU_JOINTS,
    HEAD_INDEX,
    ROOT_HIP_LEFT,
    ROOT_HIP_RIGHT,
    TORSO_SHOULDER_LEFT,
    TORSO_SHOULDER_RIGHT,
)


def test_ntu25_joint_count():
    assert len(JOINT_NAMES_NTU_25) == 25


def test_toyota13_joint_count():
    assert len(JOINT_NAMES_TOYOTA_13) == 13


def test_toyota_to_ntu_map_length():
    assert len(TOYOTA_TO_NTU_25_MAP) == 13


def test_toyota_to_ntu_map_indices_in_range():
    for idx in TOYOTA_TO_NTU_25_MAP:
        assert 0 <= idx < 25, f"Index {idx} out of range [0, 24]"


def test_toyota_to_ntu_map_all_unique():
    assert len(set(TOYOTA_TO_NTU_25_MAP)) == 13, "Mapped NTU indices must be unique"


def test_visible_ntu_joints_size():
    assert len(VISIBLE_NTU_JOINTS) == 13


def test_visible_ntu_joints_matches_map():
    assert VISIBLE_NTU_JOINTS == set(TOYOTA_TO_NTU_25_MAP)


def test_missing_ntu_joints_size():
    # 25 total − 13 visible = 12 missing
    assert len(MISSING_NTU_JOINTS) == 12


def test_visible_and_missing_partition_all_joints():
    all_joints = set(range(25))
    visible = VISIBLE_NTU_JOINTS
    missing = set(MISSING_NTU_JOINTS)
    assert visible | missing == all_joints
    assert visible & missing == set(), "No joint should be both visible and missing"


def test_ntu_edges_indices_in_range():
    for src, dst in NTU_EDGES:
        assert 0 <= src < 25, f"Edge source {src} out of range"
        assert 0 <= dst < 25, f"Edge dest {dst} out of range"


def test_ntu_edges_no_self_loops():
    for src, dst in NTU_EDGES:
        assert src != dst, f"Self-loop at joint {src}"


def test_root_and_head_constants_in_range():
    for name, val in [
        ("ROOT_HIP_LEFT", ROOT_HIP_LEFT),
        ("ROOT_HIP_RIGHT", ROOT_HIP_RIGHT),
        ("TORSO_SHOULDER_LEFT", TORSO_SHOULDER_LEFT),
        ("TORSO_SHOULDER_RIGHT", TORSO_SHOULDER_RIGHT),
        ("HEAD_INDEX", HEAD_INDEX),
    ]:
        assert 0 <= val < 25, f"{name}={val} out of range"
