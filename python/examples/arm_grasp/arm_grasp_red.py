import argparse
import time
import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import geometry_pb2, arm_command_pb2, robot_command_pb2, trajectory_pb2, gripper_command_pb2
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder

def detect_red(cv_image):
    """HSV based red detection for Version 5.1.1 testing."""
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    # Red wrap-around masks
    mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask = mask1 + mask2
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 50:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

def run_red_search(config):
    # Setup
    sdk = bosdyn.client.create_standard_sdk('RedSearch511')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        
        # 1. Stand
        robot.logger.info("Standing...")
        command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
        time.sleep(2.0)

        # 2. Build Arm + Gripper Command (5.1.1 Syntax)
        robot.logger.info("Opening claw and gazing at floor...")

        # Arm Pose: 0.7m forward, 0.3m down, wrist pitched 90 deg down
        # In 5.1.1, the helper for this is reliable
        arm_command = RobotCommandBuilder.arm_pose_command(
            0.7, 0, -0.3, 0.707, 0, 0.707, 0, VISION_FRAME_NAME, 3.0
        )

        # Gripper Command: 1.0 is fully open. 
        # In 5.1.1 ScalarTrajectoryPoint uses 'value'
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

        # Combine into a Synchronized Command
        # This is the 5.1.1 way to ensure arm and gripper move together
        synchro_command = robot_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command.synchronized_command.arm_command,
            gripper_command=gripper_command.synchronized_command.gripper_command
        )
        
        full_command = robot_command_pb2.RobotCommand(synchronized_command=synchro_command)
        command_client.robot_command(full_command)
        
        # Give the arm time to reach the position
        time.sleep(4.0)

        # 3. Search Loop
        robot.logger.info("Search loop active. CV window should appear.")
        try:
            while True:
                # Capture
                responses = image_client.get_image_from_sources([config.image_source])
                img = cv2.imdecode(np.frombuffer(responses[0].shot.image.data, dtype=np.uint8), -1)
                
                coords = detect_red(img)
                if coords:
                    print(f"RED DETECTED at: {coords}")
                    cv2.circle(img, coords, 20, (0, 255, 0), 2)
                
                cv2.imshow("Spot Hand Camera (5.1.1)", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass

        robot.power_off(cut_immediately=False)

def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', default='hand_color_image')
    options = parser.parse_args()
    run_red_search(options)

if __name__ == '__main__':
    main()