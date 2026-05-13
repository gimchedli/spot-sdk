import sys
import time
import bosdyn.client
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.api import world_object_pb2, geometry_pb2
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME

def main(argv):
    sdk = bosdyn.client.create_standard_sdk('AprilTagGrasper')
    robot = sdk.create_robot(argv[0])
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()

    world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    SCALE_FACTOR = 0.06 / 0.146

    try:
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            print("Powering on...")
            robot.power_on()
            time.sleep(1)
            
            print("Standing up...")
            command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
            time.sleep(2)

            while True:
                response = world_object_client.list_world_objects(
                    object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG]
                )
                
                if not response.world_objects:
                    print("Searching for tag...")
                    time.sleep(0.5)
                    continue

                for obj in response.world_objects:
                    try:
                        vision_tform_fiducial = get_a_tform_b(
                            obj.transforms_snapshot, 
                            VISION_FRAME_NAME, 
                            obj.apriltag_properties.frame_name_fiducial
                        )
                    except Exception:
                        continue

                    cx = vision_tform_fiducial.x * SCALE_FACTOR
                    cy = vision_tform_fiducial.y * SCALE_FACTOR
                    cz = vision_tform_fiducial.z * SCALE_FACTOR

                    # --- NAVIGATION LOGIC ---
                    # Distance on the floor (X and Y plane)
                    flat_dist = (cx**2 + cy**2)**0.5
                    
                    # If target is more than 0.8m away, walk forward
                    if flat_dist > 0.8:
                        print(f"Target is {flat_dist:.2f}m away. Walking forward...")
                        # Velocity command: v_x=0.5 m/s, v_y=0, v_rot=0
                        # We send the command for a short duration (0.5s)
                        walk_cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.5, v_y=0, v_rot=0)
                        command_client.robot_command(walk_cmd, end_time_secs=time.time() + 0.5)
                        time.sleep(0.5)
                        continue # Re-scan after moving

                    print(f"Target in range! (Dist: {flat_dist:.2f}m)")

                    # --- GRASPING LOGIC ---
                    print("Unstowing arm...")
                    command_client.robot_command(RobotCommandBuilder.arm_ready_command())
                    time.sleep(1.5)

                    print(f"Moving arm to: X={cx:.2f}, Y={cy:.2f}, Z={cz:.2f}")
                    arm_pose_cmd = RobotCommandBuilder.arm_pose_command(
                        cx, cy, cz, 1, 0, 0, 0, VISION_FRAME_NAME, 0.6
                    )
                    
                    cmd_id = command_client.robot_command(arm_pose_cmd)
                    block_until_arm_arrives(command_client, cmd_id, timeout_sec=5.0)
                    
                    print("Closing gripper...")
                    command_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(0.0))
                    time.sleep(1)
                    
                    # Optional: Stow the arm after grasping
                    print("Stowing arm...")
                    command_client.robot_command(RobotCommandBuilder.arm_stow_command())
                    time.sleep(2)

                    print("Task Complete.")
                    return

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <ROBOT_IP>")
        sys.exit(1)
    main(sys.argv[1:])