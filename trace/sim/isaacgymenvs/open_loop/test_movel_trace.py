import numpy as np

from isaacgymenvs.open_loop.movel_trace import point_to_polyline_distance, summarize


def test_point_to_polyline_uses_segments_not_only_vertices():
    distance = point_to_polyline_distance([[0.5, 0.001]], [[0.0, 0.0], [1.0, 0.0]])
    assert np.allclose(distance, [.001])


def test_exact_measured_trace_passes_geometry_and_orientation():
    x = np.linspace(0.0, 0.1, 101)
    reference = np.column_stack((x, np.zeros_like(x), np.full_like(x, .02),
                                 np.full_like(x, np.pi), np.zeros_like(x), np.zeros_like(x)))
    samples = [{"time_s": float(i / 100), "tcp_pose": pose.tolist(),
                "tcp_speed": [0.1, 0, 0, 0, 0, 0]}
               for i, pose in enumerate(reference)]
    result = summarize(reference, samples, teacher_duration_s=1.0)
    assert result["geometry_passed"]
    assert result["perpendicular_passed"]
    assert np.isclose(result["path_length_ratio_percent"], 100)
    assert result["max_actual_to_reference_xy_error_mm"] < 1e-9


def test_offset_trace_fails_millimetre_criterion():
    reference = [[0, 0, .02, np.pi, 0, 0], [.1, 0, .02, np.pi, 0, 0]]
    samples = [{"time_s": 0, "tcp_pose": [0, .003, .02, np.pi, 0, 0],
                "tcp_speed": [0, 0, 0, 0, 0, 0]},
               {"time_s": 1, "tcp_pose": [.1, .003, .02, np.pi, 0, 0],
                "tcp_speed": [0, 0, 0, 0, 0, 0]}]
    assert not summarize(reference, samples)["geometry_passed"]
