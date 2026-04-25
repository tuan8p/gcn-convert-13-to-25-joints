"""Shared constants for SSR 13->25 joints."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ssr_gcn_kaggle.yaml"

ETRI_ELDERLY_PREFIX = "etri_activity3d_elderly_"
TOYOTA_PREFIX = "toyota_smarthome_"

JOINT_NAMES_NTU_25: list[str] = [
    "SpineBase",
    "SpineMid",
    "Neck",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
    "SpineShoulder",
    "HandTipLeft",
    "ThumbLeft",
    "HandTipRight",
    "ThumbRight",
]

JOINT_NAMES_TOYOTA_13: list[str] = [
    "Rank",
    "Lank",
    "Rkne",
    "Lkne",
    "Rhip",
    "Lhip",
    "Rwri",
    "Lwri",
    "Relb",
    "Lelb",
    "Rsho",
    "Lsho",
    "Head",
]

NTU_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 20),
    (2, 20),
    (3, 2),
    (4, 20),
    (5, 4),
    (6, 5),
    (7, 6),
    (8, 20),
    (9, 8),
    (10, 9),
    (11, 10),
    (12, 0),
    (13, 12),
    (14, 13),
    (15, 14),
    (16, 0),
    (17, 16),
    (18, 17),
    (19, 18),
    (21, 7),
    (22, 7),
    (23, 11),
    (24, 11),
]

TOYOTA_TO_NTU_25_MAP: list[int] = [
    18,
    14,
    17,
    13,
    16,
    12,
    10,
    6,
    9,
    5,
    8,
    4,
    3,
]

# 7 phần tử đầu = 7 tên đầu trong JOINT_NAMES_TOYOTA_13
# (Rank, Lank, Rkne, Lkne, Rhip, Lhip, Rwri) -> NTU: ankles/knees/hips + Rwri
TOYOTA_FIRST7_NTU_INDICES: list[int] = TOYOTA_TO_NTU_25_MAP[:7]

VISIBLE_NTU_JOINTS = set(TOYOTA_TO_NTU_25_MAP)
MISSING_NTU_JOINTS = [idx for idx in range(25) if idx not in VISIBLE_NTU_JOINTS]

# 6 phần còn lại trong tập 13 (Lwri, elbows, shoulders, head)
TOYOTA_REST6_VISIBLE_NTU_INDICES: list[int] = [
    j for j in TOYOTA_TO_NTU_25_MAP if j not in TOYOTA_FIRST7_NTU_INDICES
]

# Khớp thiếu ở phần đuôi/tay: lỗi thường cao; torô (0,1,2) tách riêng để trọng số
TORSO_MISSING_NTU: frozenset[int] = frozenset({0, 1, 2})
EXTREMITY_HEAVY_MISSING_NTU: frozenset[int] = frozenset(
    {j for j in MISSING_NTU_JOINTS if j not in (0, 1, 2)}
)

ROOT_HIP_LEFT = 12
ROOT_HIP_RIGHT = 16
TORSO_SHOULDER_LEFT = 4
TORSO_SHOULDER_RIGHT = 8
HEAD_INDEX = 3
