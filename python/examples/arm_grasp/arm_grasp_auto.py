# Copyright (c) 2026 Boston Dynamics, Inc.  All rights reserved.
#
# This example shows how to automate Spot arm grasping from an image coordinate,
# using the manipulation API and optional depth inspection.

import argparse
import sys
import time

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient, pixel_format_to_numpy_type, pixel_to_camera_space
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import ResourceAlreadyClaimedError


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


def find_target_pixel_by_color(image, lower_hsv=(0, 100, 100), upper_hsv=(10, 255, 255)):
    if image.ndim != 3:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    if lower[0] <= upper[0]:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # Hue wrap-around case, e.g. red spans 170-180 and 0-10.
        mask1 = cv2.inRange(hsv, np.array((0, lower[1], lower[2]), dtype=np.uint8), upper)
        mask2 = cv2.inRange(hsv, lower, np.array((180, upper[1], upper[2]), dtype=np.uint8))
        mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None
    return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])


def get_marker_dictionary_id(dict_name):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError('OpenCV marker dictionary not available: %s' % dict_name)
    return getattr(cv2.aruco, dict_name)


def find_target_pixel_by_aruco(image, dict_name='DICT_APRILTAG_36H11'):
    if image.ndim != 3:
        return None
    try:
        dict_id = get_marker_dictionary_id(dict_name)
    except ValueError:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    parameters = (
        cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, 'DetectorParameters_create')
        else cv2.aruco.DetectorParameters())
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if corners is None or ids is None or len(corners) == 0:
        return None
    # Choose the largest detected marker by area.
    def marker_area(c):
        return cv2.contourArea(c.reshape((4, 2)).astype(np.int32))
    best_index = max(range(len(corners)), key=lambda i: marker_area(corners[i]))
    marker_corners = corners[best_index].reshape((4, 2))
    cx = int(marker_corners[:, 0].mean())
    cy = int(marker_corners[:, 1].mean())
    return cx, cy

WALK_GAZE_MODE_MAP = {
    'AUTO_GAZE': manipulation_api_pb2.PICK_AUTO_GAZE,
    'AUTO_WALK_AND_GAZE': manipulation_api_pb2.PICK_AUTO_WALK_AND_GAZE,
    'NO_AUTO_WALK_OR_GAZE': manipulation_api_pb2.PICK_NO_AUTO_WALK_OR_GAZE,
    'PLAN_ONLY': manipulation_api_pb2.PICK_PLAN_ONLY,
}


def get_depth_at_pixel(image_proto, x, y):
    if image_proto.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        return None
    depth_image = image_proto_to_array(image_proto)
    if y < 0 or y >= depth_image.shape[0] or x < 0 or x >= depth_image.shape[1]:
        return None
    depth_mm = float(depth_image[y, x])
    if depth_mm <= 0:
        return None
    return depth_mm / 1000.0


def add_grasp_constraint(config, grasp, robot_state_client):
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)
        else:
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)
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

    lease_keep_alive = None
    try:
        lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
            lease_client, must_acquire=True, return_at_exit=True)
    except ResourceAlreadyClaimedError:
        if config.take_lease:
            robot.logger.warning('Body lease already claimed; taking lease.')
            lease_client.take(resource='body')
            lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
                lease_client, must_acquire=False, return_at_exit=True)
        else:
            raise

    with lease_keep_alive:
        robot.logger.info('Powering on robot...')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'

        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)

        # Continuous image stream polling instead of a single snapshot.
        # The original snapshot code is commented out below so it is not lost.
        # robot.logger.info('Getting an image from: %s', config.image_source)
        # image_responses = image_client.get_image_from_sources([config.image_source])
        # if len(image_responses) != 1:
        #     raise RuntimeError(f'Got invalid number of images: {len(image_responses)}')
        #
        # image = image_responses[0]
        # img_array = image_proto_to_array(image)

        image = None
        img_array = None
        target_x = config.target_x
        target_y = config.target_y
        detection_method = 'Manual' if config.target_x is not None and config.target_y is not None else 'Default'

        stream_frame = 0
        while True:
            robot.logger.info('Polling stream frame %d from: %s', stream_frame + 1, config.image_source)
            image_responses = image_client.get_image_from_sources([config.image_source])
            if len(image_responses) != 1:
                robot.logger.warning('Stream frame %d: got %d images from source %s',
                                   stream_frame + 1, len(image_responses), config.image_source)
                time.sleep(0.1)
                continue

            image = image_responses[0]
            img_array = image_proto_to_array(image)
            stream_frame += 1
            robot.logger.info('Received stream frame %d from: %s', stream_frame, config.image_source)

            if config.target_x is not None and config.target_y is not None:
                target_x = config.target_x
                target_y = config.target_y
                detection_method = 'Manual'
                break

            robot.logger.info('Attempting automatic target detection...')
            detected = find_target_pixel_by_aruco(img_array, dict_name=config.apriltag_dict)
            if detected is not None:
                target_x, target_y = detected
                detection_method = config.apriltag_dict
                robot.logger.info('Using marker target at pixel: %d, %d (dict=%s)',
                                  target_x, target_y, config.apriltag_dict)
                break

            robot.logger.info('No marker target found in current frame, trying color detection.')
            detected = find_target_pixel_by_color(
                img_array, lower_hsv=config.color_lower_hsv,
                upper_hsv=config.color_upper_hsv)
            if detected is not None:
                target_x, target_y = detected
                detection_method = 'Color'
                robot.logger.info('Using color target at pixel: %d, %d', target_x, target_y)
                break

            if config.show_image:
                display = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if img_array.ndim == 3 else img_array
                cv2.circle(display, (img_array.shape[1] // 2, img_array.shape[0] // 2), 12, (255, 255, 0), 1)
                cv2.putText(display, 'Streaming... waiting for target', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow('Target Image', display)
                cv2.waitKey(1)

            time.sleep(0.1)

        if img_array is None:
            raise RuntimeError('Failed to acquire image from stream.')

        robot.logger.info('Final target pixel: %d, %d (method=%s)', target_x, target_y, detection_method)

        if config.show_image:
            display = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if img_array.ndim == 3 else img_array
            cv2.circle(display, (target_x, target_y), 12, (0, 255, 0), 2)
            cv2.putText(display, f'Target: {detection_method}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Target Image', display)
            cv2.waitKey(500)

        depth_m = get_depth_at_pixel(image, target_x, target_y)
        if depth_m is not None:
            robot.logger.info('Distance to target: %.3f m', depth_m)
            print(f'Distance to target: {depth_m:.3f} m')
            if image.source.pinhole is not None:
                cam_point = pixel_to_camera_space(image.source, target_x, target_y, depth_m)
                robot.logger.info('Camera-space target: x=%.3f y=%.3f z=%.3f', *cam_point)
            if depth_m < 1.0:
                robot.logger.info('Target is under 1 meter; disabling walking for grasp attempt.')
                grasp_mode = manipulation_api_pb2.PICK_AUTO_GAZE
            else:
                grasp_mode = WALK_GAZE_MODE_MAP.get(config.walk_gaze_mode,
                                                   manipulation_api_pb2.PICK_AUTO_GAZE)
        else:
            robot.logger.info('No valid depth at target pixel; using configured walk/gaze mode.')
            grasp_mode = WALK_GAZE_MODE_MAP.get(config.walk_gaze_mode,
                                               manipulation_api_pb2.PICK_AUTO_GAZE)

        target_pixel = geometry_pb2.Vec2(x=target_x, y=target_y)
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=target_pixel,
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)
        grasp.walk_gaze_mode = grasp_mode
        robot.logger.info('Using walk/gaze mode: %s', config.walk_gaze_mode)

        add_grasp_constraint(config, grasp, robot_state_client)

        robot.logger.info('Waiting 4 seconds before grasp/move attempt...')
        time.sleep(4)

        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        while True:
            time.sleep(0.25)
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)
            state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(
                response.current_state)
            robot.logger.info('Current state: %s', state_name)
            if response.current_state in (
                    manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                    manipulation_api_pb2.MANIP_STATE_GRASP_FAILED):
                break

        robot.logger.info('Finished grasp.')
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), 'Robot power off failed.'


def parse_arguments():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', default='hand_color_image',
                        help='Image source to use for automatic grasp target selection.')
    parser.add_argument('--target-x', type=int, default=None,
                        help='Pixel x coordinate to grasp. Defaults to image center.')
    parser.add_argument('--target-y', type=int, default=None,
                        help='Pixel y coordinate to grasp. Defaults to image center.')
    parser.add_argument('--show-image', action='store_true',
                        help='Show the selected image and target pixel briefly.')
    parser.add_argument('--apriltag-dict', default='DICT_APRILTAG_36H11',
                        help='AprilTag/ArUco dictionary name to use for marker detection.')
    parser.add_argument('--walk-gaze-mode', default='AUTO_GAZE',
                        choices=list(WALK_GAZE_MODE_MAP),
                        help='Walk/gaze behavior for the grasp attempt.')
    parser.add_argument('--color-lower-hsv', nargs=3, type=int,
                        default=[0, 100, 100], help='Lower HSV bound for color detection.')
    parser.add_argument('--color-upper-hsv', nargs=3, type=int,
                        default=[10, 255, 255], help='Upper HSV bound for color detection.')
    parser.add_argument('--take-lease', action='store_true',
                        help='Take the body lease if it is already held by another client.')
    parser.add_argument('-t', '--force-top-down-grasp', action='store_true',
                        help='Force a top-down grasp.')
    parser.add_argument('-f', '--force-horizontal-grasp', action='store_true',
                        help='Force a horizontal grasp.')
    parser.add_argument('-r', '--force-45-angle-grasp', action='store_true',
                        help='Force a 45-degree angled grasp.')
    parser.add_argument('-s', '--force-squeeze-grasp', action='store_true',
                        help='Force a squeeze grasp.')
    return parser.parse_args()


def main():
    options = parse_arguments()
    num = sum((options.force_top_down_grasp, options.force_horizontal_grasp,
               options.force_45_angle_grasp, options.force_squeeze_grasp))
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



# Commented out: Basic object detection using OpenCV contours.
# This function finds the center of the largest contour in the image,
# assuming it's the target object (e.g., a drone).
# Uncomment to enable automatic object detection.
# def detect_object_center(img_array):
#     if img_array.ndim != 3:
#         return None  # Only works on color images
#     gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None
#     largest_contour = max(contours, key=cv2.contourArea)
#     M = cv2.moments(largest_contour)
#     if M['m00'] == 0:
#         return None
#     cx = int(M['m10'] / M['m00'])
#     cy = int(M['m01'] / M['m00'])
#     return cx, cy

# arm_grasp_auto.py 10.22.41.210
