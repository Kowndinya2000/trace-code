"""Utility: send the UR5e to its home joint configuration via ur_rtde moveJ.

Also contains (commented) a workspace-boundary walk and a traj04.txt waypoint
replay experiment — an early precursor of the open-loop trajectory executor.
"""
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import numpy as np
from robotiq_gripper import RobotiqGripper


if __name__ == "__main__":
    tool_vel = 0.1
    tool_acc = 0.1 
    joint_vel = 0.5
    joint_acc = 0.5
    _ip = os.environ.get("TRACE_ROBOT_IP", "192.168.1.102") 
    min_z_clearance = 0.018
    
    # Connect to the UR5e robot
    rtde_c = RTDEControl(_ip)
    rtde_r = RTDEReceive(_ip, use_upper_range_registers=False) 

    gripper = RobotiqGripper(os.environ.get("TRACE_ROBOT_IP", "192.168.1.102"), 63352)
    gripper.connect()
    
    # Lift the gripper to ensure clearance over scene objects

    print(rtde_r.getActualQ())
    # q = [
    #     73.96,
    #     -114.66,
    #     117.57,
    #     -92.90,
    #     -89.78,
    #     -196.15
    # ]

    # Standard 2F-85 gripper finger pads (4.5 cms)
    # q = [
    #     -19.32+90,
    #     -100.29,
    #     147.48,
    #     -137.18,
    #     -89.72,
    #     160.73
    # ]

    # Extended 2F-85 gripper finger pads (27 cms)
    q = [
        70.78,
        -118.35,
        120.67,
        -92.32,
        -89.50,
        160.50
    ]
    joint_home = [x*(np.pi/180) for x in q]  # robot will go here before starting the calibration
    
    # rtde_c.servoJ(joint_home, 1.2, 0.25, 0.008, 0.1, 300) # Very fast servoing 
    
    # rtde_c.servoJ(joint_home, 1.2, 0.25, 0.02, 0.2, 100) # smooth servoing to the home position
    # gripper.close_and_wait_for_pos(gripper._max_speed / 3, gripper._max_force / 150)
    # gripper.open(gripper._max_speed / 3, gripper._max_force / 150) 
    rtde_c.moveJ(joint_home, 0.2, 0.2)
    # gripper.close_and_wait_for_pos(gripper._max_speed / 3, gripper._max_force / 150)
    current_pose = rtde_r.getActualTCPPose()
    print("Current Pose: ", current_pose)
    exit(0)

    current_pose = rtde_r.getActualTCPPose()
    current_pose_lift = current_pose.copy()
    current_pose_lift[2] = min_z_clearance + 0.07
    # rtde_c.moveL(current_pose_lift, tool_vel, tool_acc)

    # Define the home position with a lift
    home_pose_lift = np.array([
                        (-0.235+0.213)/2, 
                        -0.190, # -0.30, 
                        min_z_clearance + 0.07, 
                        -3.14, 
                        -0.0, 
                        -0.0
                        ])
    
    # rtde_c.moveL(home_pose_lift, tool_vel, tool_acc)

    # Move to the home position at the minimum z clearance
    home_pose = home_pose_lift.copy()
    home_pose[2] = min_z_clearance
    # rtde_c.moveL(home_pose, tool_vel, tool_acc)
    # exit(0)


    real_workspace = np.asarray(
        [
            [-0.424, 0.224],
            [-0.076, -0.724],
            [min_z_clearance, min_z_clearance]
        ]
    )

    workspace_boundary_poses = np.array(
        [
            [real_workspace[0][0], real_workspace[1][0], min_z_clearance, -3.14, -0.0, -0.0],
            [real_workspace[0][0], real_workspace[1][1], min_z_clearance, -3.14, -0.0, -0.0],
            [real_workspace[0][1], real_workspace[1][1], min_z_clearance, -3.14, -0.0, -0.0],
            [real_workspace[0][1], real_workspace[1][0], min_z_clearance, -3.14, -0.0, -0.0]
        ]
    )
    for ee_pose in workspace_boundary_poses[0:3]:
        # rtde_c.moveL(ee_pose, tool_vel, tool_acc)
        pass

    read_poses = open("traj04.txt", "r")
    for line in read_poses:
        pose_values = line.strip().split()
        pose = [float(value) for value in pose_values]
        ee_pose = [pose[1], -pose[0], min_z_clearance, 3.14, -0.0, -0.0]
        rtde_c.moveL(ee_pose, tool_vel, tool_acc)

    current_pose = rtde_r.getActualTCPPose()
    current_pose_lift = current_pose.copy()
    current_pose_lift[2] = 0.08
    rtde_c.moveL(current_pose_lift, tool_vel, tool_acc)

    target_pose = rtde_r.getActualTCPPose()
    target_pose[0] = -0.005
    target_pose[1] = -0.600
    target_pose[3] = 2.305
    target_pose[4] = 2.133
    target_pose[5] = -0.004
    rtde_c.moveL(target_pose, tool_vel, tool_acc)
    gripper.open(gripper._max_speed, gripper._max_force) 


    go_down = rtde_r.getActualTCPPose()
    go_down[2] = min_z_clearance
    rtde_c.moveL(go_down, tool_vel, tool_acc)

    gripper.close_and_wait_for_pos(gripper._max_speed, gripper._max_force / 150)

    lift_pose = rtde_r.getActualTCPPose()
    lift_pose[2] = 0.085
    rtde_c.moveL(lift_pose, tool_vel, tool_acc)