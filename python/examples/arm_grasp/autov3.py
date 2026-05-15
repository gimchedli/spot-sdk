# Copyright (c) 2026 Boston Dynamics, Inc.  All rights reserved.
#
# This example shows how to automate Spot arm grasping from an image coordinate,
# using the manipulation API and optional depth inspection.
# Upgraded to include multi-camera scanning, dynamic tracking, size estimation, live video, and frontal grasps.

import argparse
import sys
import time
import math
from pathlib import Path

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, get_a_tform_b, math_helpers
from bosdyn.client.image import ImageClient, pixel_format_to_numpy_type, pixel_to_camera_space
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import ResourceAlreadyClaimedError

ALL_CAMERAS = [
    'hand_color_image', 'frontleft_fisheye_image', 'frontright_fisheye_image',
    'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image'
]

def verify_estop(robot):
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        raise Exception('Robot is estopped. Please clear the estop before running this example.')

def image_proto_to_array(image_proto):
    if image_proto.shot.image.format == image_pb2.Image.FORMAT_RAW:
        dtype = pixel_format_to_numpy_type(image_proto.shot.image.pixel_format)
        arr = np.frombuffer(image_proto.shot.image.data, dtype=dtype)
        return arr.reshape(image_proto.shot.image.rows, image_proto.shot.image.cols)
    return cv2.imdecode(np.frombuffer(image_proto.shot.image.data, dtype=np.uint8), -1)

def get_arm_grasp_folder():
    return Path(__file__).resolve().parent

def save_image_to_arm_grasp_folder(image, filename):
    output_path = get_arm_grasp_folder() / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path

def calculate_target_size(bbox, depth_m, camera_intrinsics):
    if not camera_intrinsics or depth_m is None or bbox is None:
        return None, None
    x_focal = camera_intrinsics.intrinsics.focal_length.x
    y_focal = camera_intrinsics.intrinsics.focal_length.y
    _, _, w, h = bbox
    width_m = (w * depth_m) / x_focal
    height_m = (h * depth_m) / y_focal
    return width_m * 100, height_m * 100 # Return in cm

def find_target_pixel_by_color(image, lower_hsv=(0, 100, 100), upper_hsv=(10, 255, 255)):
    if image.ndim != 3:
        return None, None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    if lower[0] <= upper[0]:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        mask1 = cv2.inRange(hsv, np.array((0, lower[1], lower[2]), dtype=np.uint8), upper)
        mask2 = cv2.inRange(hsv, lower, np.array((180, upper[1], upper[2]), dtype=np.uint8))
        mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None, None
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None, None
    bbox = cv2.boundingRect(largest)
    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])), bbox

def find_target_pixel_by_red(image):
    if image.ndim != 3:
        return None, None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array((0, 100, 100), dtype=np.uint8), np.array((10, 255, 255), dtype=np.uint8)
    lower2, upper2 = np.array((170, 100, 100), dtype=np.uint8), np.array((180, 255, 255), dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None, None
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None, None
    bbox = cv2.boundingRect(largest)
    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])), bbox

def get_marker_dictionary_id(dict_name):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError('OpenCV marker dictionary not available: %s' % dict_name)
    return getattr(cv2.aruco, dict_name)

def is_arm_camera_source(source_name):
    return source_name and (source_name.startswith('hand_') or source_name.startswith('hand'))

def assert_arm_camera_source(image_source_name):
    if not is_arm_camera_source(image_source_name):
        raise RuntimeError(f'Image source is not an arm camera source: {image_source_name}.')

def find_target_pixel_by_aruco(image, dict_name='DICT_APRILTAG_36H11'):
    # ArUco detection works on grayscale, so we don't block ndim != 3 here
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    try: dict_id = get_marker_dictionary_id(dict_name)
    except ValueError: return None, None
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    parameters = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, 'DetectorParameters_create') else cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
    if corners is None or ids is None or len(corners) == 0: return None, None
    best_index = max(range(len(corners)), key=lambda i: cv2.contourArea(corners[i].reshape((4, 2)).astype(np.int32)))
    marker_corners = corners[best_index].reshape((4, 2))
    cx, cy = int(marker_corners[:, 0].mean()), int(marker_corners[:, 1].mean())
    bbox = cv2.boundingRect(corners[best_index].reshape((4, 2)).astype(np.int32))
    return (cx, cy), bbox

WALK_GAZE_MODE_MAP = {
    'AUTO_GAZE': manipulation_api_pb2.PICK_AUTO_GAZE,
    'AUTO_WALK_AND_GAZE': manipulation_api_pb2.PICK_AUTO_WALK_AND_GAZE,
    'NO_AUTO_WALK_OR_GAZE': manipulation_api_pb2.PICK_NO_AUTO_WALK_OR_GAZE,
    'PLAN_ONLY': manipulation_api_pb2.PICK_PLAN_ONLY,
}

def get_depth_at_pixel(image_proto, x, y):
    if not image_proto or image_proto.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16: return None
    depth_image = image_proto_to_array(image_proto)
    if y < 0 or y >= depth_image.shape[0] or x < 0 or x >= depth_image.shape[1]: return None
    depth_mm = float(depth_image[y, x])
    if depth_mm <= 0: return None
    return depth_mm / 1000.0

def add_grasp_constraint(config, grasp, robot_state_client):
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp or config.force_frontal_grasp
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=-1)
        elif config.force_horizontal_grasp:
            axis_on_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=1)
        elif config.force_frontal_grasp:
            axis_on_gripper = geometry_pb2.Vec3(x=0, y=0, z=1)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=1)

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align)
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif config.force_45_angle_grasp:
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif config.force_squeeze_grasp:
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()

def detect_target_in_image(img_array, config):
    if config.target_x is not None and config.target_y is not None:
        return (config.target_x, config.target_y), 'Manual', None

    detected, bbox = find_target_pixel_by_aruco(img_array, dict_name=config.apriltag_dict)
    if detected: return detected, config.apriltag_dict, bbox

    # Color detection only works if the image has 3 channels (is not grayscale)
    if img_array.ndim == 3:
        detected, bbox = find_target_pixel_by_red(img_array)
        if detected: return detected, 'Red', bbox

        detected, bbox = find_target_pixel_by_color(img_array, lower_hsv=config.color_lower_hsv, upper_hsv=config.color_upper_hsv)
        if detected: return detected, 'Color', bbox

    return None, None, None

def run_auto_grasp(config):
    bosdyn.client.util.setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk('ArmAutoGraspClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'
    verify_estop(robot)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    lease_keep_alive = None
    try:
        lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
    except ResourceAlreadyClaimedError:
        if config.take_lease:
            robot.logger.warning('Body lease already claimed; taking lease.')
            lease_client.take(resource='body')
            lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=False, return_at_exit=True)
        else:
            raise

    with lease_keep_alive:
        robot.logger.info('Powering on robot...')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'

        robot.logger.info('Commanding robot to stand...')
        blocking_stand(command_client, timeout_sec=10)

        robot.logger.info('Recording starting position in Vision frame...')
        initial_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(initial_state.kinematic_state.transforms_snapshot)
        start_x, start_y, start_yaw = vision_T_body.x, vision_T_body.y, vision_T_body.rot.to_yaw()

        robot.logger.info('Unstowing arm to a ready position to see the target...')
        command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        time.sleep(2.0)

        robot.logger.info('Opening gripper jaws to clear camera view...')
        gripper_open_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        command_client.robot_command(gripper_open_cmd)
        time.sleep(1.0)

        robot.logger.info("Scanning cameras for target...")
        active_cam = config.image_source
        display_cam_idx = 0
        target_dist = 0.75
        K_yaw, K_x = 0.002, 0.5
        
        while True:
            display_cam_name = ALL_CAMERAS[display_cam_idx]
            
            # Request cameras needed for logic AND the UI display
            reqs = [active_cam]
            if active_cam == 'hand_color_image':
                reqs.append('hand_depth_in_hand_color_frame')
            if config.show_image and display_cam_name not in reqs:
                reqs.append(display_cam_name)
                
            img_responses = image_client.get_image_from_sources(reqs)
            if not img_responses: continue
            
            # Map responses by name for easy extraction
            resp_dict = {img.source.name: img for img in img_responses}
            color_img = resp_dict.get(active_cam)
            depth_img = resp_dict.get('hand_depth_in_hand_color_frame')
            ui_img = resp_dict.get(display_cam_name)
            
            img_array = image_proto_to_array(color_img)
            center, detection_method, bbox = detect_target_in_image(img_array, config)
            
            # If nothing in primary view, do a background scan of all cameras
            if center is None and active_cam == config.image_source:
                scan_reqs = [bosdyn.client.image.build_image_request(src) for src in ALL_CAMERAS]
                images = image_client.get_image(scan_reqs)
                found = False
                for img in images:
                    arr = image_proto_to_array(img)
                    c, m, _ = detect_target_in_image(arr, config)
                    if c:
                        active_cam = img.source.name
                        found = True
                        robot.logger.info(f"Target found in {active_cam} via {m}")
                        break
                
                if not found:
                    command_client.robot_command(RobotCommandBuilder.synchro_velocity_command(0, 0, 0), end_time_secs=time.time() + 1.0)
                    
            if center:
                target_x, target_y = center
                depth_m = get_depth_at_pixel(depth_img, target_x, target_y) if depth_img else None
                
                img_center_x = color_img.source.cols / 2.0
                v_rot = np.clip((img_center_x - target_x) * K_yaw, -0.5, 0.5)
                
                if depth_m:
                    w_cm, h_cm = calculate_target_size(bbox, depth_m, color_img.source.pinhole)
                    if w_cm:
                        robot.logger.info(f"Size: {w_cm:.1f}x{h_cm:.1f}cm | Dist: {depth_m:.2f}m")
                    
                    if depth_m <= target_dist and is_arm_camera_source(active_cam):
                        command_client.robot_command(RobotCommandBuilder.synchro_velocity_command(0, 0, 0), end_time_secs=time.time() + 1.0)
                        break 
                        
                    v_x = np.clip((depth_m - target_dist) * K_x, -0.4, 0.4)
                else:
                    v_x = 0.0 # Just rotate to center if no depth (e.g. tracking via body camera)
                    
                vel_cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=0.0, v_rot=v_rot)
                command_client.robot_command(vel_cmd, end_time_secs=time.time() + 0.5)
                
            # If tracking via a body camera, test if it has rotated into the front hand camera's view
            if active_cam != config.image_source:
                test_hand = image_client.get_image_from_sources([config.image_source])
                if test_hand:
                    c, _, _ = detect_target_in_image(image_proto_to_array(test_hand[0]), config)
                    if c:
                        active_cam = config.image_source
                        robot.logger.info("Target rotated into hand camera view. Switching back.")

            # --- LIVE VIDEO FEED WITH CAMERA SWITCHER ---
            if config.show_image and ui_img:
                disp_arr = image_proto_to_array(ui_img)
                
                # Convert grayscale fisheye feeds to BGR so we can draw colored text/circles on them
                if disp_arr.ndim == 2:
                    disp_arr = cv2.cvtColor(disp_arr, cv2.COLOR_GRAY2BGR)
                
                # Only draw the targeting circle if we are actively looking through the tracking camera
                if display_cam_name == active_cam and center:
                    cv2.circle(disp_arr, (target_x, target_y), 12, (0, 255, 0), 2)
                    cv2.putText(disp_arr, f'Tracking: {active_cam}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    state_txt = f'Target locked in: {active_cam}' if center else f'Searching: {active_cam}'
                    cv2.putText(disp_arr, state_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw the UI helper text
                cv2.putText(disp_arr, f'View: {display_cam_name} [Press C to cycle]', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('Live Spot Camera', disp_arr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') or key == ord('C'):
                    display_cam_idx = (display_cam_idx + 1) % len(ALL_CAMERAS)
                
            time.sleep(0.1)

        # Re-acquire final high-res image and aligned depth from hand
        image_responses = image_client.get_image_from_sources([config.image_source, 'hand_depth_in_hand_color_frame'])
        image = image_responses[0]
        depth_img_final = image_responses[1] if len(image_responses) > 1 else None
        
        img_array = image_proto_to_array(image)
        center, detection_method, bbox = detect_target_in_image(img_array, config)
        if center is None: raise RuntimeError('Failed to re-acquire target from hand camera.')
        target_x, target_y = center

        robot.logger.info('Final target pixel: %d, %d (method=%s)', target_x, target_y, detection_method)
        save_image_to_arm_grasp_folder(img_array, 'snapshot_1_after_stand.jpg')
        viz_array = img_array.copy()
        cv2.circle(viz_array, (target_x, target_y), 12, (0, 255, 0), 2)
        save_image_to_arm_grasp_folder(viz_array, f'snapshot_1_target_{detection_method}.jpg')

        depth_m = get_depth_at_pixel(depth_img_final, target_x, target_y) if depth_img_final else None
        cam_point = pixel_to_camera_space(image.source, target_x, target_y, depth_m) if depth_m and image.source.pinhole else None

        # =======================================================================
        # FINGERTIP CARTESIAN GRASP LOGIC
        # =======================================================================
        if config.cube_size is not None and depth_m is not None and cam_point is not None:
            robot.logger.info(f'Cube size {config.cube_size} cm specified. Executing Cartesian fingertip grasp!')
            snapshot = image.shot.transforms_snapshot
            vision_T_camera = get_a_tform_b(snapshot, VISION_FRAME_NAME, image.shot.frame_name_image_sensor)
            camera_T_target = math_helpers.SE3Pose(cam_point[0], cam_point[1], cam_point[2], math_helpers.Quat())
            vision_T_target = vision_T_camera * camera_T_target
            
            robot_state = robot_state_client.get_robot_state()
            vision_T_hand = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot, VISION_FRAME_NAME, 'hand')

            # --- REACH OUT AND PINCH ---
            pull_back_m = 0.16 + ((config.cube_size / 2.0) / 100.0)
            hx, hy, hz = vision_T_target.x - vision_T_hand.x, vision_T_target.y - vision_T_hand.y, vision_T_target.z - vision_T_hand.z
            h_dist = math.sqrt(hx**2 + hy**2 + hz**2)
            
            new_hand_x = vision_T_target.x - (hx / h_dist) * pull_back_m
            new_hand_y = vision_T_target.y - (hy / h_dist) * pull_back_m
            new_hand_z = vision_T_target.z - (hz / h_dist) * pull_back_m
            new_vision_T_hand = math_helpers.SE3Pose(new_hand_x, new_hand_y, new_hand_z, vision_T_hand.rotation)
            
            arm_cmd = RobotCommandBuilder.arm_pose_command(
                new_vision_T_hand.x, new_vision_T_hand.y, new_vision_T_hand.z,
                new_vision_T_hand.rot.w, new_vision_T_hand.rot.x, new_vision_T_hand.rot.y, new_vision_T_hand.rot.z,
                VISION_FRAME_NAME, 4.0 
            )
            command_client.robot_command(arm_cmd)
            time.sleep(4.5) 
            
            image_responses_2 = image_client.get_image_from_sources([config.image_source])
            if len(image_responses_2) == 1:
                save_image_to_arm_grasp_folder(image_proto_to_array(image_responses_2[0]), 'snapshot_2_before_grip.jpg')
                
            close_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)
            command_client.robot_command(close_cmd)
            time.sleep(1.0)

        # =======================================================================
        # STANDARD AUTO-GRASP FALLBACK
        # =======================================================================
        else:
            grasp_mode = manipulation_api_pb2.PICK_NO_AUTO_WALK_OR_GAZE if depth_m and depth_m < 1.0 else WALK_GAZE_MODE_MAP.get(config.walk_gaze_mode, manipulation_api_pb2.PICK_AUTO_GAZE)
            target_pixel = geometry_pb2.Vec2(x=target_x, y=target_y)
            grasp = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=target_pixel, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                frame_name_image_sensor=image.shot.frame_name_image_sensor, camera_model=image.source.pinhole)
            grasp.walk_gaze_mode = grasp_mode

            add_grasp_constraint(config, grasp, robot_state_client)
            grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
            cmd_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=grasp_request)

            took_snapshot_2 = False
            while True:
                time.sleep(0.2) 
                response = manipulation_api_client.manipulation_api_feedback_command(
                    manipulation_api_feedback_request=manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_response.manipulation_cmd_id))
                
                if not took_snapshot_2 and response.current_state in (
                        manipulation_api_pb2.MANIP_STATE_GRASPING_OBJECT,
                        manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED):
                    
                    img_resp_2 = image_client.get_image_from_sources([config.image_source])
                    if len(img_resp_2) == 1:
                        save_image_to_arm_grasp_folder(image_proto_to_array(img_resp_2[0]), 'snapshot_2_before_grip.jpg')
                    took_snapshot_2 = True

                if response.current_state in (manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED):
                    break

        # =======================================================================
        # RESET AND RETURN SEQUENCE 
        # =======================================================================
        time.sleep(3.0)
        command_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0))
        time.sleep(1.5)
        command_client.robot_command(RobotCommandBuilder.arm_stow_command())
        time.sleep(3.0)

        robot.logger.info('Walking back to recorded starting position...')
        return_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=start_x, goal_y=start_y, goal_heading=start_yaw, frame_name=VISION_FRAME_NAME)
        command_client.robot_command(return_cmd, end_time_secs=time.time() + 15.0)
        time.sleep(10.0) 

        if config.show_image: cv2.destroyAllWindows()

        robot.logger.info('Experiment complete. Powering off.')
        robot.power_off(cut_immediately=False, timeout_sec=20)

def parse_arguments():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', default='hand_color_image')
    parser.add_argument('--target-x', type=int, default=None)
    parser.add_argument('--target-y', type=int, default=None)
    parser.add_argument('--show-image', action='store_true')
    parser.add_argument('--apriltag-dict', default='DICT_APRILTAG_36H11')
    parser.add_argument('--walk-gaze-mode', default='AUTO_GAZE', choices=list(WALK_GAZE_MODE_MAP))
    parser.add_argument('--color-lower-hsv', nargs=3, type=int, default=[0, 100, 100])
    parser.add_argument('--color-upper-hsv', nargs=3, type=int, default=[10, 255, 255])
    parser.add_argument('--take-lease', action='store_true')
    parser.add_argument('-t', '--force-top-down-grasp', action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp', action='store_true')
    parser.add_argument('-r', '--force-45-angle-grasp', action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp', action='store_true')
    parser.add_argument('--force-frontal-grasp', action='store_true', help='Force Spot to approach and grasp horizontally from the front.')
    parser.add_argument('--cube-size', type=float, default=None)
    return parser.parse_args()

def main():
    options = parse_arguments()
    num = sum((options.force_top_down_grasp, options.force_horizontal_grasp,
               options.force_45_angle_grasp, options.force_squeeze_grasp, options.force_frontal_grasp))
    if num > 1:
        print('Error: choose at most one grasp constraint.')
        return False
    try:
        run_auto_grasp(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)