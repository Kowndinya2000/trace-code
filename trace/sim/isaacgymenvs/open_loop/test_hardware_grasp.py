"""Self-check for the single PMBS post-checked hardware grasp gate."""

from isaacgymenvs.open_loop.real_grasp import enforce_hardware_grasp_checks


def proposal(**updates):
    result = {
        "q": 0.9,
        "network_graspable": True,
        "grasp_post_processing": "pmbs_demo",
        "x_real": 0.0,
        "y_real": -0.5,
        "surface_z_m": 0.045,
        "rotation_idx": 4,
    }
    result.update(updates)
    return result


def main():
    accepted = proposal()
    enforce_hardware_grasp_checks(accepted)
    assert accepted["graspable"]
    assert accepted["hardware_grasp_checks_passed"]

    # Legacy hand-built clearance measurements are not a second policy. PMBS
    # has already collision-masked the exact selected pixel and orientation.
    formerly_vetoed = proposal(
        graspable=False,
        jaw_clutter_px=97,
        perp_clearance_mm=25,
        target_to_concave_clearance_camera_px=2.2,
    )
    enforce_hardware_grasp_checks(formerly_vetoed)
    assert formerly_vetoed["graspable"]

    # The score is recomputed at the gate; stale metadata cannot override it.
    low_score = proposal(q=0.69, network_graspable=True)
    enforce_hardware_grasp_checks(low_score)
    assert not low_score["graspable"]
    assert "GN score below" in low_score["reject_reason"]

    missing_pose = proposal(x_real=None)
    enforce_hardware_grasp_checks(missing_pose)
    assert not missing_pose["graspable"]
    assert "no valid grasp pose" in missing_pose["reject_reason"]

    unchecked = proposal(grasp_post_processing=None)
    enforce_hardware_grasp_checks(unchecked)
    assert not unchecked["graspable"]
    assert "did not use PMBS post-checking" in unchecked["reject_reason"]

    print("PMBS post-checked hardware gate: 5 checks passed")


if __name__ == "__main__":
    main()
