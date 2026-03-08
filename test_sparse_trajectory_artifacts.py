import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("PROJECT_DIR", str(Path(__file__).resolve().parent))

from agents.cinematographer import (  # noqa: E402
    compute_sparse_frame_indices,
    interpolate_dense_trajectory,
)
from agents.critic import validate_critic_output  # noqa: E402
from lift_uv_traj_to_3d import load_uv_trajectory  # noqa: E402


@pytest.mark.parametrize(
    ("num_frames", "divisor", "expected"),
    [
        (1, 2, [0]),
        (1, 4, [0]),
        (2, 2, [0, 1]),
        (2, 4, [0, 1]),
        (3, 2, [0, 2]),
        (3, 4, [0, 2]),
    ],
)
def test_compute_sparse_frame_indices_small_cases(
    num_frames: int, divisor: int, expected: list[int]
) -> None:
    assert compute_sparse_frame_indices(num_frames, divisor) == expected


@pytest.mark.parametrize(("num_frames", "divisor", "expected_count"), [(70, 2, 35), (70, 4, 18)])
def test_compute_sparse_frame_indices_standard_case(
    num_frames: int, divisor: int, expected_count: int
) -> None:
    indices = compute_sparse_frame_indices(num_frames, divisor)
    assert indices[0] == 0
    assert indices[-1] == num_frames - 1
    assert indices == sorted(set(indices))
    assert len(indices) == expected_count


def test_interpolate_dense_trajectory_preserves_sparse_anchors() -> None:
    strided = {
        "t_0": {"position": [10.0, 20.0]},
        "t_3": {"position": [40.0, 50.0]},
        "t_6": {"position": [100.0, 200.0]},
    }

    dense = interpolate_dense_trajectory(strided, num_frames=7)

    assert dense["t_0"]["position"] == [10.0, 20.0]
    assert dense["t_3"]["position"] == [40.0, 50.0]
    assert dense["t_6"]["position"] == [100.0, 200.0]
    assert dense["t_1"]["position"] == [20.0, 30.0]
    assert dense["t_2"]["position"] == [30.0, 40.0]
    assert dense["t_4"]["position"] == [60.0, 100.0]
    assert dense["t_5"]["position"] == [80.0, 150.0]


def test_interpolate_dense_trajectory_does_not_clamp_bounds() -> None:
    strided = {
        "t_0": {"position": [-5.0, 20.0]},
        "t_2": {"position": [600.0, 700.0]},
    }

    dense = interpolate_dense_trajectory(strided, num_frames=3)

    assert dense["t_0"]["position"] == [-5.0, 20.0]
    assert dense["t_1"]["position"] == [297.5, 360.0]
    assert dense["t_2"]["position"] == [600.0, 700.0]


def test_dense_json_contract_is_compatible_with_lift_loader(tmp_path: Path) -> None:
    strided = {
        "t_0": {"position": [100.0, 120.0]},
        "t_4": {"position": [140.0, 180.0]},
    }
    dense = {
        "camera_trajectory_pixels": interpolate_dense_trajectory(
            strided,
            num_frames=5,
        )
    }

    dense_path = tmp_path / "camera_trajectory_pixels.json"
    dense_path.write_text(json.dumps(dense), encoding="utf-8")

    uv = load_uv_trajectory(dense_path)
    assert len(uv) == 5
    assert uv[0] == (100.0, 120.0)
    assert uv[-1] == (140.0, 180.0)


def test_validate_critic_output_accepts_sparse_key_feedback() -> None:
    critic_json = {
        "intent_matched": False,
        "analysis": "Sparse control at t_23 drifts left and loses the racket.",
        "actionable_feedback": "Move t_23 right by 18 px and smooth the span from t_23 to t_46.",
        "failure_modes": ["off_subject_framing"],
        "frame_level_notes": {"t_23": "Target begins to drift left."},
        "overall_quality_score": 4.0,
        "refinement_suggestions": ["maintain_subject_framing"],
    }

    validate_critic_output(critic_json, allowed_sparse_keys={"t_0", "t_23", "t_46", "t_69"})


def test_validate_critic_output_rejects_non_sparse_frame_notes() -> None:
    critic_json = {
        "intent_matched": False,
        "analysis": "Trajectory misses the subject.",
        "actionable_feedback": "Move t_23 right by 18 px.",
        "failure_modes": ["off_subject_framing"],
        "frame_level_notes": {"t_1": "This key is not sparse."},
        "overall_quality_score": 2.0,
        "refinement_suggestions": ["maintain_subject_framing"],
    }

    with pytest.raises(RuntimeError, match="sparse control keys"):
        validate_critic_output(critic_json, allowed_sparse_keys={"t_0", "t_23", "t_46", "t_69"})


def test_validate_critic_output_requires_sparse_key_reference_when_intent_false() -> None:
    critic_json = {
        "intent_matched": False,
        "analysis": "Trajectory misses the subject.",
        "actionable_feedback": "Move the camera slightly to the right.",
        "failure_modes": ["off_subject_framing"],
        "frame_level_notes": {"t_23": "Drift starts here."},
        "overall_quality_score": 2.0,
        "refinement_suggestions": ["maintain_subject_framing"],
    }

    with pytest.raises(RuntimeError, match="must reference sparse control keys"):
        validate_critic_output(critic_json, allowed_sparse_keys={"t_0", "t_23", "t_46", "t_69"})
