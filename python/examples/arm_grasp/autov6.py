# Copyright (c) 2026 Boston Dynamics, Inc. All rights reserved.
#
# Spot AprilTag grasp helper:
# - Searches all cameras for AprilTags.
# - Does NOT rotate Spot body during search.
# - Uses body height changes to handle vertical visibility problems.
# - If a fisheye camera sees the target, points the ARM/HAND CAMERA that way.
# - Keeps one live camera view for monitoring.
# - Final grasp is attempted only from hand_color_image.
# - Default grasp is Cartesian fingertip grasp using hand depth.
# - Always releases object before stowing/returning to start.
# - Returns to recorded start pose by default.

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
    math_helpers,
)
from bosdyn.client.image import ImageClient, pixel_format_to_numpy_type, pixel_to_camera_space
from bosdyn.client.lease import ResourceAlreadyClaimedError
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient


WINDOW_NAME = "Spot AprilTag Search"

ALL_CAMERAS = [
    "hand_color_image",
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
]

HAND_CAMERA = "hand_color_image"
HAND_DEPTH = "hand_depth_in_hand_color_frame"


WALK_GAZE_MODE_MAP = {
    "NO_AUTO_WALK_OR_GAZE": manipulation_api_pb2.PICK_NO_AUTO_WALK_OR_GAZE,
    "AUTO_GAZE": manipulation_api_pb2.PICK_AUTO_GAZE,
    "AUTO_WALK_AND_GAZE": manipulation_api_pb2.PICK_AUTO_WALK_AND_GAZE,
    "PLAN_ONLY": manipulation_api_pb2.PICK_PLAN_ONLY,
}


def verify_estop(robot):
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        raise RuntimeError("Robot is estopped. Clear estop before running this script.")


def get_output_folder():
    return Path(__file__).resolve().parent


def save_image(image, filename):
    output_path = get_output_folder() / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def image_proto_to_array(image_proto):
    img = image_proto.shot.image

    if img.format == image_pb2.Image.FORMAT_RAW:
        dtype = pixel_format_to_numpy_type(img.pixel_format)
        arr = np.frombuffer(img.data, dtype=dtype)

        if img.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            rgb = arr.reshape(img.rows, img.cols, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if hasattr(image_pb2.Image, "PIXEL_FORMAT_RGBA_U8"):
            if img.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                rgba = arr.reshape(img.rows, img.cols, 4)
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

        return arr.reshape(img.rows, img.cols)

    decoded = cv2.imdecode(np.frombuffer(img.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if decoded is None:
        raise RuntimeError(f"Could not decode image from source: {image_proto.source.name}")

    return decoded


def ensure_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image.copy()


def safe_get_images_from_sources(image_client, sources, robot_logger=None):
    unique_sources = []

    for src in sources:
        if src and src not in unique_sources:
            unique_sources.append(src)

    try:
        return image_client.get_image_from_sources(unique_sources)
    except Exception as exc:
        if robot_logger:
            robot_logger.warning(
                "Batch image request failed. Falling back to individual requests: %s",
                exc,
            )

    responses = []

    for src in unique_sources:
        try:
            single = image_client.get_image_from_sources([src])
            responses.extend(single)
        except Exception as exc:
            if robot_logger:
                robot_logger.debug("Image source failed: %s, error=%s", src, exc)

    return responses


def get_marker_dictionary_id(dict_name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is missing. Install OpenCV contrib package, for example: "
            "pip install opencv-contrib-python"
        )

    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"OpenCV marker dictionary not available: {dict_name}")

    return getattr(cv2.aruco, dict_name)


def find_target_pixel_by_apriltag(image, dict_name="DICT_APRILTAG_36H11"):
    if image is None:
        return None

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return None

    try:
        dict_id = get_marker_dictionary_id(dict_name)
    except ValueError:
        return None

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    parameters = (
        cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "DetectorParameters_create")
        else cv2.aruco.DetectorParameters()
    )

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if corners is None or ids is None or len(corners) == 0:
        return None

    def marker_area(corner_set):
        pts = corner_set.reshape((4, 2)).astype(np.int32)
        return cv2.contourArea(pts)

    best_index = max(range(len(corners)), key=lambda i: marker_area(corners[i]))
    marker_corners = corners[best_index].reshape((4, 2))

    cx = int(marker_corners[:, 0].mean())
    cy = int(marker_corners[:, 1].mean())

    bbox = cv2.boundingRect(marker_corners.astype(np.int32))
    marker_id = int(ids[best_index][0])
    area = float(marker_area(corners[best_index]))

    return {
        "center": (cx, cy),
        "bbox": bbox,
        "id": marker_id,
        "area": area,
        "corners": marker_corners,
    }


def get_depth_image_m(image_proto):
    if image_proto is None:
        return None

    if image_proto.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        return None

    depth_image_mm = image_proto_to_array(image_proto).astype(np.float32)
    depth_image_m = depth_image_mm / 1000.0
    depth_image_m[depth_image_m <= 0.0] = np.nan

    return depth_image_m


def get_depth_at_pixel(image_proto, x, y):
    depth_image_m = get_depth_image_m(image_proto)

    if depth_image_m is None:
        return None

    if y < 0 or y >= depth_image_m.shape[0] or x < 0 or x >= depth_image_m.shape[1]:
        return None

    depth_m = float(depth_image_m[y, x])

    if not np.isfinite(depth_m) or depth_m <= 0:
        return None

    return depth_m


def get_median_depth_around_pixel(image_proto, x, y, radius_px):
    depth_image_m = get_depth_image_m(image_proto)

    if depth_image_m is None:
        return None

    rows, cols = depth_image_m.shape[:2]

    x0 = max(0, x - radius_px)
    x1 = min(cols, x + radius_px + 1)
    y0 = max(0, y - radius_px)
    y1 = min(rows, y + radius_px + 1)

    crop = depth_image_m[y0:y1, x0:x1]
    valid = crop[np.isfinite(crop)]

    if valid.size == 0:
        return get_depth_at_pixel(image_proto, x, y)

    return float(np.median(valid))


def clamp(value, low, high):
    return max(low, min(high, value))


def send_zero_velocity(command_client):
    stop_cmd = RobotCommandBuilder.synchro_velocity_command(
        v_x=0.0,
        v_y=0.0,
        v_rot=0.0,
    )
    command_client.robot_command(stop_cmd, end_time_secs=time.time() + 0.5)


def send_body_height(command_client, body_height_m, settle_sec=0.8):
    stand_cmd = RobotCommandBuilder.synchro_stand_command(body_height=body_height_m)
    command_client.robot_command(stand_cmd)
    time.sleep(settle_sec)


def command_arm_pose(command_client, pose, frame_name, duration_sec):
    cmd = RobotCommandBuilder.arm_pose_command(
        pose.x,
        pose.y,
        pose.z,
        pose.rot.w,
        pose.rot.x,
        pose.rot.y,
        pose.rot.z,
        frame_name,
        duration_sec,
    )

    command_client.robot_command(cmd)
    time.sleep(duration_sec + 0.25)


def draw_detection(display, detection, source_name, body_height):
    display = ensure_bgr(display)

    if detection:
        x, y = detection["center"]
        cv2.circle(display, (x, y), 12, (0, 255, 0), 2)

        corners = detection.get("corners")
        if corners is not None:
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        cv2.putText(
            display,
            f"Tag ID: {detection['id']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            display,
            "No tag in this view",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        display,
        f"View: {source_name} | C=cycle | Q/ESC=stop | body_height={body_height:+.2f}m",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        display,
        "Search movement: body height + arm steering. No body rotation.",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )

    return display


def scan_all_cameras_for_tags(image_client, config, robot_logger=None):
    responses = safe_get_images_from_sources(
        image_client,
        ALL_CAMERAS,
        robot_logger=robot_logger,
    )

    detections = {}

    for response in responses:
        try:
            arr = image_proto_to_array(response)
            detection = find_target_pixel_by_apriltag(arr, config.apriltag_dict)

            if detection:
                source_name = response.source.name
                detection["source"] = source_name
                detection["image_response"] = response
                detection["image_array"] = arr
                detection["rows"] = arr.shape[0]
                detection["cols"] = arr.shape[1]
                detections[source_name] = detection

        except Exception as exc:
            if robot_logger:
                robot_logger.debug("Detection failed for image source: %s", exc)

    return detections


def choose_best_detection(detections):
    if HAND_CAMERA in detections:
        return detections[HAND_CAMERA]

    if not detections:
        return None

    return max(detections.values(), key=lambda d: d["area"])


def update_body_height_from_detection(command_client, detection, current_height, config, robot_logger):
    _, tag_y = detection["center"]
    rows = detection["rows"]

    image_center_y = rows / 2.0
    vertical_error_norm = (tag_y - image_center_y) / image_center_y

    if abs(vertical_error_norm) < config.vertical_deadband:
        return current_height, False

    delta = -config.body_height_gain * vertical_error_norm

    new_height = clamp(
        current_height + delta,
        config.body_height_min,
        config.body_height_max,
    )

    if abs(new_height - current_height) < 0.01:
        return current_height, False

    robot_logger.info(
        "Vertical correction from %s: tag_y=%d, error=%.2f, body_height %.2f -> %.2f",
        detection["source"],
        tag_y,
        vertical_error_norm,
        current_height,
        new_height,
    )

    send_body_height(command_client, new_height, settle_sec=config.body_height_settle_sec)
    return new_height, True


def get_arm_gaze_target_from_camera_detection(detection, config):
    """
    Returns a rough gaze target in BODY frame.

    Body frame convention:
    x = forward
    y = left
    z = up

    This is intentionally approximate. Fisheye cameras are used as direction hints,
    not as final grasp cameras.
    """
    source = detection["source"]
    tag_x, tag_y = detection["center"]
    rows = detection["rows"]
    cols = detection["cols"]

    x = config.arm_gaze_distance_m
    y = 0.0

    if source == "frontleft_fisheye_image":
        x = config.arm_gaze_distance_m
        y = config.arm_gaze_side_offset_m

    elif source == "frontright_fisheye_image":
        x = config.arm_gaze_distance_m
        y = -config.arm_gaze_side_offset_m

    elif source == "left_fisheye_image":
        x = config.arm_gaze_forward_offset_m
        y = config.arm_gaze_distance_m

    elif source == "right_fisheye_image":
        x = config.arm_gaze_forward_offset_m
        y = -config.arm_gaze_distance_m

    elif source == "back_fisheye_image":
        x = -config.arm_gaze_distance_m
        y = 0.0

    elif source == HAND_CAMERA:
        x = config.arm_gaze_distance_m
        y = 0.0

    horizontal_error = (tag_x - cols / 2.0) / (cols / 2.0)
    vertical_error = (tag_y - rows / 2.0) / (rows / 2.0)

    if source in ("frontleft_fisheye_image", "frontright_fisheye_image", HAND_CAMERA):
        y += -config.arm_gaze_pixel_lateral_gain_m * horizontal_error

    z = config.arm_gaze_height_m - config.arm_gaze_pixel_vertical_gain_m * vertical_error
    z = clamp(z, config.arm_gaze_min_height_m, config.arm_gaze_max_height_m)

    return x, y, z


def steer_arm_towards_detection(command_client, detection, config, robot_logger):
    if not config.enable_arm_camera_steering:
        return False

    if detection is None:
        return False

    if detection["source"] == HAND_CAMERA:
        return False

    x, y, z = get_arm_gaze_target_from_camera_detection(detection, config)

    robot_logger.info(
        "Arm steering hint from %s: commanding hand gaze toward BODY target x=%.2f y=%.2f z=%.2f",
        detection["source"],
        x,
        y,
        z,
    )

    try:
        gaze_cmd = RobotCommandBuilder.arm_gaze_command(
            x,
            y,
            z,
            BODY_FRAME_NAME,
        )

        command_client.robot_command(
            gaze_cmd,
            end_time_secs=time.time() + config.arm_gaze_command_duration_sec,
        )

        time.sleep(config.arm_gaze_settle_sec)
        return True

    except Exception as exc:
        robot_logger.warning("Arm gaze command failed: %s", exc)
        return False


def wait_until_hand_camera_sees_tag(image_client, command_client, config, robot_logger):
    current_height = clamp(
        config.body_height_start,
        config.body_height_min,
        config.body_height_max,
    )

    send_body_height(command_client, current_height, settle_sec=1.0)

    display_idx = 0
    sweep_heights = config.body_height_sweep
    sweep_idx = 0
    last_sweep_time = 0.0
    last_arm_steer_time = 0.0

    start_time = time.time()

    while True:
        if time.time() - start_time > config.search_timeout_sec:
            raise RuntimeError(
                f"Timed out after {config.search_timeout_sec}s. "
                "No stable AprilTag target found in hand camera."
            )

        detections = scan_all_cameras_for_tags(
            image_client,
            config,
            robot_logger=robot_logger,
        )

        best = choose_best_detection(detections)

        if best:
            robot_logger.info(
                "Tag detected in %s: id=%s, center=%s, area=%.1f",
                best["source"],
                best["id"],
                best["center"],
                best["area"],
            )

            current_height, _ = update_body_height_from_detection(
                command_client,
                best,
                current_height,
                config,
                robot_logger,
            )

            if best["source"] != HAND_CAMERA:
                now = time.time()
                if now - last_arm_steer_time >= config.arm_steering_interval_sec:
                    steer_arm_towards_detection(
                        command_client,
                        best,
                        config,
                        robot_logger,
                    )
                    last_arm_steer_time = now

            hand_detection = detections.get(HAND_CAMERA)

            if hand_detection:
                _, tag_y = hand_detection["center"]
                rows = hand_detection["rows"]
                vertical_error_norm = abs((tag_y - rows / 2.0) / (rows / 2.0))

                if vertical_error_norm <= config.hand_vertical_acceptance:
                    robot_logger.info(
                        "Hand camera has acceptable target view: center=%s, vertical_error=%.2f",
                        hand_detection["center"],
                        vertical_error_norm,
                    )
                    return hand_detection

        else:
            now = time.time()

            if now - last_sweep_time > config.body_height_sweep_interval_sec:
                current_height = sweep_heights[sweep_idx % len(sweep_heights)]
                sweep_idx += 1
                last_sweep_time = now

                robot_logger.info(
                    "No tag in any camera. Sweeping body height to %.2f m",
                    current_height,
                )

                send_body_height(
                    command_client,
                    current_height,
                    settle_sec=config.body_height_settle_sec,
                )

        if config.show_image:
            display_source = ALL_CAMERAS[display_idx % len(ALL_CAMERAS)]

            responses = safe_get_images_from_sources(
                image_client,
                [display_source],
                robot_logger=robot_logger,
            )

            if responses:
                display_response = responses[0]
                display_arr = image_proto_to_array(display_response)

                if display_source in detections:
                    display_detection = detections[display_source]
                else:
                    display_detection = find_target_pixel_by_apriltag(
                        display_arr,
                        config.apriltag_dict,
                    )

                display = draw_detection(
                    display_arr.copy(),
                    display_detection,
                    display_source,
                    current_height,
                )

                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("c"), ord("C")):
                    display_idx = (display_idx + 1) % len(ALL_CAMERAS)

                if key in (ord("q"), ord("Q"), 27):
                    robot_logger.warning("User requested stop from camera window.")
                    send_zero_velocity(command_client)
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt("Stopped by user from camera window.")

        time.sleep(config.scan_sleep_sec)


def get_selected_grasp_orientation(config):
    legacy_flags = [
        config.force_top_down_grasp,
        config.force_horizontal_grasp,
        config.force_45_angle_grasp,
        config.force_squeeze_grasp,
    ]

    if sum(legacy_flags) > 1:
        raise RuntimeError("Choose at most one grasp constraint.")

    if config.force_top_down_grasp:
        return "top_down"

    if config.force_horizontal_grasp:
        return "horizontal"

    if config.force_45_angle_grasp:
        return "angle_45"

    if config.force_squeeze_grasp:
        return "squeeze"

    return config.grasp_orientation


def add_grasp_constraint(config, grasp, robot_state_client):
    orientation = get_selected_grasp_orientation(config)

    if orientation == "unconstrained":
        return

    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if orientation == "top_down":
        axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
        axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=-1)

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper
        )
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align
        )
        constraint.vector_alignment_with_tolerance.threshold_radians = (
            config.grasp_constraint_tolerance_rad
        )

    elif orientation == "horizontal":
        axis_on_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
        axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=1)

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper
        )
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align
        )
        constraint.vector_alignment_with_tolerance.threshold_radians = (
            config.grasp_constraint_tolerance_rad
        )

    elif orientation == "frontal":
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )

        body_Q_grasp = math_helpers.Quat.from_roll(math.radians(config.wrist_roll_deg))
        vision_Q_grasp = vision_T_body.rot * body_Q_grasp

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
            vision_Q_grasp.to_proto()
        )
        constraint.rotation_with_tolerance.threshold_radians = (
            config.grasp_constraint_tolerance_rad
        )

    elif orientation == "angle_45":
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )

        body_Q_grasp = math_helpers.Quat.from_pitch(math.radians(45.0))
        vision_Q_grasp = vision_T_body.rot * body_Q_grasp

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
            vision_Q_grasp.to_proto()
        )
        constraint.rotation_with_tolerance.threshold_radians = (
            config.grasp_constraint_tolerance_rad
        )

    elif orientation == "squeeze":
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()

    else:
        raise RuntimeError(f"Unknown grasp orientation: {orientation}")


def reacquire_hand_target(image_client, config, robot_logger):
    responses = safe_get_images_from_sources(
        image_client,
        [HAND_CAMERA, HAND_DEPTH],
        robot_logger=robot_logger,
    )

    by_source = {r.source.name: r for r in responses}

    hand_image = by_source.get(HAND_CAMERA)
    depth_image = by_source.get(HAND_DEPTH)

    if hand_image is None:
        raise RuntimeError("Failed to reacquire hand camera image before grasp.")

    hand_arr = image_proto_to_array(hand_image)
    detection = find_target_pixel_by_apriltag(hand_arr, config.apriltag_dict)

    if not detection:
        raise RuntimeError("AprilTag disappeared from hand camera before grasp.")

    detection["source"] = HAND_CAMERA
    detection["image_response"] = hand_image
    detection["image_array"] = hand_arr
    detection["rows"] = hand_arr.shape[0]
    detection["cols"] = hand_arr.shape[1]

    x, y = detection["center"]

    depth_m = None

    if depth_image is not None:
        depth_m = get_median_depth_around_pixel(
            depth_image,
            x,
            y,
            radius_px=config.depth_window_px,
        )

    robot_logger.info(
        "Final hand target: id=%s, pixel=(%d,%d), depth=%s",
        detection["id"],
        x,
        y,
        f"{depth_m:.3f}m" if depth_m else "not available",
    )

    return detection, hand_image, depth_image, hand_arr, depth_m


def make_annotated_final_image(final_arr, final_detection, depth_m, config):
    annotated_final = ensure_bgr(final_arr)

    fx, fy = final_detection["center"]
    cv2.circle(annotated_final, (fx, fy), 12, (0, 255, 0), 2)

    corners = final_detection.get("corners")
    if corners is not None:
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_final, [pts], True, (0, 255, 0), 2)

    depth_text = f"{depth_m:.3f}m" if depth_m else "no depth"

    cv2.putText(
        annotated_final,
        f"Final target: ({fx},{fy}) depth={depth_text}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated_final,
        f"fingertip_standoff={config.fingertip_standoff_m:.3f}m",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
    )

    return annotated_final


def execute_cartesian_fingertip_grasp(
    command_client,
    robot_state_client,
    image_response,
    detection,
    depth_m,
    config,
    robot_logger,
):
    if depth_m is None:
        raise RuntimeError(
            "Cannot execute Cartesian fingertip grasp because no valid hand depth was available."
        )

    target_x, target_y = detection["center"]

    cam_point = pixel_to_camera_space(
        image_response.source,
        target_x,
        target_y,
        depth_m,
    )

    snapshot = image_response.shot.transforms_snapshot

    vision_T_camera = get_a_tform_b(
        snapshot,
        VISION_FRAME_NAME,
        image_response.shot.frame_name_image_sensor,
    )

    camera_T_target = math_helpers.SE3Pose(
        cam_point[0],
        cam_point[1],
        cam_point[2],
        math_helpers.Quat(),
    )

    vision_T_target = vision_T_camera * camera_T_target

    robot_state = robot_state_client.get_robot_state()
    vision_T_hand = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        VISION_FRAME_NAME,
        "hand",
    )

    hand_position = np.array(
        [vision_T_hand.x, vision_T_hand.y, vision_T_hand.z],
        dtype=np.float64,
    )

    tag_position = np.array(
        [vision_T_target.x, vision_T_target.y, vision_T_target.z],
        dtype=np.float64,
    )

    approach_vector = tag_position - hand_position
    approach_distance = float(np.linalg.norm(approach_vector))

    if approach_distance < 0.05:
        raise RuntimeError(
            f"Hand is too close to target to compute stable approach vector: {approach_distance:.3f}m"
        )

    approach_unit = approach_vector / approach_distance

    contact_position = tag_position + approach_unit * config.target_depth_offset_m

    pregrasp_position = contact_position - approach_unit * (
        config.fingertip_standoff_m + config.pregrasp_extra_m
    )

    grasp_position = contact_position - approach_unit * config.fingertip_standoff_m

    wrist_roll = math_helpers.Quat.from_roll(math.radians(config.wrist_roll_deg))
    desired_rotation = vision_T_hand.rot * wrist_roll

    pregrasp_pose = math_helpers.SE3Pose(
        float(pregrasp_position[0]),
        float(pregrasp_position[1]),
        float(pregrasp_position[2]),
        desired_rotation,
    )

    grasp_pose = math_helpers.SE3Pose(
        float(grasp_position[0]),
        float(grasp_position[1]),
        float(grasp_position[2]),
        desired_rotation,
    )

    robot_logger.info(
        "Cartesian fingertip grasp target: vision=(%.3f, %.3f, %.3f), depth=%.3fm",
        vision_T_target.x,
        vision_T_target.y,
        vision_T_target.z,
        depth_m,
    )

    robot_logger.info(
        "Approach distance=%.3fm, fingertip_standoff=%.3fm, target_depth_offset=%.3fm",
        approach_distance,
        config.fingertip_standoff_m,
        config.target_depth_offset_m,
    )

    robot_logger.info(
        "Pregrasp pose: x=%.3f y=%.3f z=%.3f",
        pregrasp_pose.x,
        pregrasp_pose.y,
        pregrasp_pose.z,
    )

    robot_logger.info(
        "Grasp pose: x=%.3f y=%.3f z=%.3f",
        grasp_pose.x,
        grasp_pose.y,
        grasp_pose.z,
    )

    robot_logger.info("Opening gripper before Cartesian approach.")
    command_client.robot_command(
        RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
    )
    time.sleep(0.7)

    robot_logger.info("Moving to Cartesian pregrasp pose.")
    command_arm_pose(
        command_client,
        pregrasp_pose,
        VISION_FRAME_NAME,
        config.cartesian_pregrasp_sec,
    )

    robot_logger.info("Advancing to fingertip grasp pose.")
    command_arm_pose(
        command_client,
        grasp_pose,
        VISION_FRAME_NAME,
        config.cartesian_grasp_sec,
    )

    robot_logger.info("Closing gripper for fingertip grasp.")
    command_client.robot_command(
        RobotCommandBuilder.claw_gripper_open_fraction_command(config.close_fraction)
    )
    time.sleep(config.close_wait_sec)

    return True


def execute_manipulation_api_grasp(
    manipulation_api_client,
    robot_state_client,
    command_client,
    image_response,
    detection,
    config,
    robot_logger,
):
    target_x, target_y = detection["center"]
    target_pixel = geometry_pb2.Vec2(x=target_x, y=target_y)

    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=target_pixel,
        transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
        frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
        camera_model=image_response.source.pinhole,
    )

    if config.allow_auto_walk_gaze:
        grasp.walk_gaze_mode = manipulation_api_pb2.PICK_AUTO_WALK_AND_GAZE
        robot_logger.warning(
            "AUTO_WALK_AND_GAZE is enabled. Spot may move or rotate during the grasp planner."
        )
    else:
        grasp.walk_gaze_mode = manipulation_api_pb2.PICK_NO_AUTO_WALK_OR_GAZE
        robot_logger.info("Using PICK_NO_AUTO_WALK_OR_GAZE to avoid search/planner rotation.")

    selected_orientation = get_selected_grasp_orientation(config)

    robot_logger.info(
        "Using manipulation API grasp orientation: %s, wrist_roll_deg=%.1f, tolerance_rad=%.2f",
        selected_orientation,
        config.wrist_roll_deg,
        config.grasp_constraint_tolerance_rad,
    )

    add_grasp_constraint(config, grasp, robot_state_client)

    grasp_request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp
    )

    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request
    )

    robot_logger.info("Manipulation command sent. Waiting for feedback...")

    start_time = time.time()
    last_state = None

    while True:
        if time.time() - start_time > config.grasp_timeout_sec:
            robot_logger.warning("Grasp timed out after %.1f seconds.", config.grasp_timeout_sec)
            return manipulation_api_pb2.MANIP_STATE_GRASP_FAILED

        if config.show_image:
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                robot_logger.warning("User requested stop during grasp feedback.")
                send_zero_velocity(command_client)
                raise KeyboardInterrupt("Stopped by user during grasp feedback.")

        time.sleep(0.25)

        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id
        )

        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request
        )

        current_state = response.current_state

        if current_state != last_state:
            try:
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(current_state)
            except Exception:
                state_name = str(current_state)

            robot_logger.info("Manipulation state: %s", state_name)
            last_state = current_state

        if current_state in (
            manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
        ):
            return current_state


def release_gripper(command_client, robot_logger, wait_sec=1.0):
    robot_logger.info("Opening gripper before stowing/returning.")
    command_client.robot_command(
        RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
    )
    time.sleep(wait_sec)


def run(config):
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk("SpotAprilTagFingertipGrasp")
    robot = sdk.create_robot(config.hostname)

    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm."
    verify_estop(robot)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    try:
        lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
            lease_client,
            must_acquire=True,
            return_at_exit=True,
        )
    except ResourceAlreadyClaimedError:
        if config.take_lease:
            robot.logger.warning("Body lease already claimed. Taking lease.")
            lease_client.take(resource="body")
            lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
                lease_client,
                must_acquire=False,
                return_at_exit=True,
            )
        else:
            raise

    with lease_keep_alive:
        robot.logger.info("Powering on robot.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."

        robot.logger.info("Standing.")
        blocking_stand(command_client, timeout_sec=10)

        initial_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(
            initial_state.kinematic_state.transforms_snapshot
        )

        start_x = vision_T_body.x
        start_y = vision_T_body.y
        start_yaw = vision_T_body.rot.to_yaw()

        robot.logger.info(
            "Recorded start pose: x=%.3f, y=%.3f, yaw=%.3f",
            start_x,
            start_y,
            start_yaw,
        )

        robot.logger.info("Moving arm to ready position.")
        command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        time.sleep(2.0)

        robot.logger.info("Opening gripper.")
        command_client.robot_command(
            RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        )
        time.sleep(1.0)

        robot.logger.info(
            "Searching for AprilTag using body-height movement and arm steering. No body rotation."
        )

        try:
            hand_detection = wait_until_hand_camera_sees_tag(
                image_client,
                command_client,
                config,
                robot.logger,
            )
        except KeyboardInterrupt:
            robot.logger.warning("Interrupted by user. Stopping motion, stowing arm, and powering off.")
            send_zero_velocity(command_client)
            command_client.robot_command(RobotCommandBuilder.arm_stow_command())
            time.sleep(2.0)

            if config.show_image:
                cv2.destroyAllWindows()

            if config.power_off:
                robot.power_off(cut_immediately=False, timeout_sec=20)

            return

        clean_img = hand_detection["image_array"]
        save_image(clean_img, "snapshot_1_hand_camera_clean.jpg")

        annotated = ensure_bgr(clean_img)
        x, y = hand_detection["center"]
        cv2.circle(annotated, (x, y), 12, (0, 255, 0), 2)
        save_image(annotated, "snapshot_1_hand_camera_target.jpg")

        robot.logger.info("Reacquiring final hand image and depth before grasp.")

        final_detection, final_image, final_depth_image, final_arr, depth_m = reacquire_hand_target(
            image_client,
            config,
            robot.logger,
        )

        annotated_final = make_annotated_final_image(
            final_arr,
            final_detection,
            depth_m,
            config,
        )

        save_image(annotated_final, "snapshot_2_final_before_grasp.jpg")

        if config.show_image:
            cv2.imshow(WINDOW_NAME, annotated_final)
            cv2.waitKey(500)

        grasp_success = False

        try:
            if config.use_cartesian_fingertip_grasp:
                robot.logger.info("Executing Cartesian fingertip grasp.")
                grasp_success = execute_cartesian_fingertip_grasp(
                    command_client,
                    robot_state_client,
                    final_image,
                    final_detection,
                    depth_m,
                    config,
                    robot.logger,
                )
            else:
                robot.logger.info("Executing Manipulation API grasp.")
                result = execute_manipulation_api_grasp(
                    manipulation_api_client,
                    robot_state_client,
                    command_client,
                    final_image,
                    final_detection,
                    config,
                    robot.logger,
                )

                grasp_success = result == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED

        except KeyboardInterrupt:
            robot.logger.warning("Interrupted during grasp.")
            send_zero_velocity(command_client)

        except Exception as exc:
            robot.logger.exception("Grasp execution failed: %s", exc)

        if grasp_success:
            robot.logger.info("Grasp sequence completed.")
        else:
            robot.logger.warning("Grasp sequence failed or was interrupted.")

        robot.logger.info("Holding for %.1f seconds.", config.hold_sec)
        time.sleep(config.hold_sec)

        release_gripper(command_client, robot.logger, wait_sec=1.0)

        robot.logger.info("Stowing arm.")
        command_client.robot_command(RobotCommandBuilder.arm_stow_command())
        time.sleep(3.0)

        if config.return_to_start:
            robot.logger.info("Returning to recorded start pose.")
            return_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=start_x,
                goal_y=start_y,
                goal_heading=start_yaw,
                frame_name=VISION_FRAME_NAME,
            )
            command_client.robot_command(
                return_cmd,
                end_time_secs=time.time() + config.return_timeout_sec,
            )
            time.sleep(min(config.return_timeout_sec, 10.0))

        if config.show_image:
            cv2.destroyAllWindows()

        if config.power_off:
            robot.logger.info("Powering off.")
            robot.power_off(cut_immediately=False, timeout_sec=20)


def parse_arguments():
    parser = argparse.ArgumentParser()

    bosdyn.client.util.add_base_arguments(parser)

    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Show live camera view. C cycles view. Q/ESC stops and exits.",
    )

    parser.add_argument(
        "--apriltag-dict",
        default="DICT_APRILTAG_36H11",
        help="OpenCV AprilTag/ArUco dictionary name.",
    )

    parser.add_argument(
        "--take-lease",
        action="store_true",
        help="Take body lease if already claimed.",
    )

    parser.add_argument(
        "--search-timeout-sec",
        type=float,
        default=120.0,
        help="Maximum time to search before failing.",
    )

    parser.add_argument(
        "--scan-sleep-sec",
        type=float,
        default=0.10,
        help="Sleep between scan iterations.",
    )

    parser.add_argument(
        "--body-height-start",
        type=float,
        default=0.0,
        help="Starting body height relative to nominal stand height, meters.",
    )

    parser.add_argument(
        "--body-height-min",
        type=float,
        default=-0.18,
        help="Minimum body height relative to nominal stand height, meters.",
    )

    parser.add_argument(
        "--body-height-max",
        type=float,
        default=0.18,
        help="Maximum body height relative to nominal stand height, meters.",
    )

    parser.add_argument(
        "--body-height-gain",
        type=float,
        default=0.08,
        help="Body-height correction gain from image vertical error.",
    )

    parser.add_argument(
        "--vertical-deadband",
        type=float,
        default=0.12,
        help="Ignore body-height corrections inside this normalized vertical error.",
    )

    parser.add_argument(
        "--hand-vertical-acceptance",
        type=float,
        default=0.35,
        help="Accept hand-camera tag view if vertical error is below this value.",
    )

    parser.add_argument(
        "--body-height-settle-sec",
        type=float,
        default=0.8,
        help="Settle time after body-height command.",
    )

    parser.add_argument(
        "--body-height-sweep",
        nargs="+",
        type=float,
        default=[0.0, 0.10, -0.10, 0.18, -0.18],
        help="Body-height sweep sequence used when no camera sees the tag.",
    )

    parser.add_argument(
        "--body-height-sweep-interval-sec",
        type=float,
        default=1.2,
        help="Time between body-height sweep positions.",
    )

    parser.set_defaults(enable_arm_camera_steering=True)

    parser.add_argument(
        "--enable-arm-camera-steering",
        dest="enable_arm_camera_steering",
        action="store_true",
        help="Point the arm/hand camera toward the fisheye camera direction when a fisheye sees the tag. Enabled by default.",
    )

    parser.add_argument(
        "--no-arm-camera-steering",
        dest="enable_arm_camera_steering",
        action="store_false",
        help="Disable arm steering from fisheye detections.",
    )

    parser.add_argument(
        "--arm-steering-interval-sec",
        type=float,
        default=1.5,
        help="Minimum time between arm steering commands.",
    )

    parser.add_argument(
        "--arm-gaze-command-duration-sec",
        type=float,
        default=2.0,
        help="Command expiry duration for arm gaze command.",
    )

    parser.add_argument(
        "--arm-gaze-settle-sec",
        type=float,
        default=0.6,
        help="Settle time after sending arm gaze command.",
    )

    parser.add_argument(
        "--arm-gaze-distance-m",
        type=float,
        default=1.0,
        help="Rough gaze distance used for arm steering target in body frame.",
    )

    parser.add_argument(
        "--arm-gaze-height-m",
        type=float,
        default=0.05,
        help="Base gaze height used for arm steering target in body frame.",
    )

    parser.add_argument(
        "--arm-gaze-min-height-m",
        type=float,
        default=-0.35,
        help="Minimum gaze height used for arm steering target in body frame.",
    )

    parser.add_argument(
        "--arm-gaze-max-height-m",
        type=float,
        default=0.45,
        help="Maximum gaze height used for arm steering target in body frame.",
    )

    parser.add_argument(
        "--arm-gaze-side-offset-m",
        type=float,
        default=0.35,
        help="Left/right offset used for front-left and front-right camera steering.",
    )

    parser.add_argument(
        "--arm-gaze-forward-offset-m",
        type=float,
        default=0.20,
        help="Forward offset used when steering toward left/right fisheye cameras.",
    )

    parser.add_argument(
        "--arm-gaze-pixel-lateral-gain-m",
        type=float,
        default=0.20,
        help="Small lateral correction from tag x-position in image.",
    )

    parser.add_argument(
        "--arm-gaze-pixel-vertical-gain-m",
        type=float,
        default=0.25,
        help="Small vertical correction from tag y-position in image.",
    )

    parser.set_defaults(use_cartesian_fingertip_grasp=True)

    parser.add_argument(
        "--use-cartesian-fingertip-grasp",
        dest="use_cartesian_fingertip_grasp",
        action="store_true",
        help="Use explicit depth-based Cartesian fingertip grasp. Enabled by default.",
    )

    parser.add_argument(
        "--use-manipulation-api-grasp",
        dest="use_cartesian_fingertip_grasp",
        action="store_false",
        help="Use Boston Dynamics PickObjectInImage instead of Cartesian fingertip grasp.",
    )

    parser.add_argument(
        "--fingertip-standoff-m",
        type=float,
        default=0.16,
        help=(
            "Distance from hand frame target pose to fingertip contact point. "
            "Increase if palm hits. Decrease if fingers close in air."
        ),
    )

    parser.add_argument(
        "--pregrasp-extra-m",
        type=float,
        default=0.05,
        help="Extra pullback before final fingertip approach.",
    )

    parser.add_argument(
        "--target-depth-offset-m",
        type=float,
        default=0.0,
        help=(
            "Positive moves desired contact point behind the AprilTag along the approach vector. "
            "Useful if the tag is on the front face but the grasp point is slightly behind it."
        ),
    )

    parser.add_argument(
        "--depth-window-px",
        type=int,
        default=5,
        help="Median depth window radius around AprilTag center.",
    )

    parser.add_argument(
        "--cartesian-pregrasp-sec",
        type=float,
        default=3.0,
        help="Seconds for arm move to pregrasp pose.",
    )

    parser.add_argument(
        "--cartesian-grasp-sec",
        type=float,
        default=2.0,
        help="Seconds for arm move from pregrasp to final fingertip pose.",
    )

    parser.add_argument(
        "--close-fraction",
        type=float,
        default=0.0,
        help="Gripper open fraction after reaching fingertip pose. 0.0 is fully closed.",
    )

    parser.add_argument(
        "--close-wait-sec",
        type=float,
        default=1.0,
        help="Wait after closing gripper.",
    )

    parser.add_argument(
        "--allow-auto-walk-gaze",
        action="store_true",
        help="Allow manipulation planner to walk/gaze. Only used with --use-manipulation-api-grasp.",
    )

    parser.add_argument(
        "--grasp-orientation",
        default="frontal",
        choices=[
            "frontal",
            "top_down",
            "horizontal",
            "angle_45",
            "squeeze",
            "unconstrained",
        ],
        help="Manipulation API orientation. Only used with --use-manipulation-api-grasp.",
    )

    parser.add_argument(
        "--wrist-roll-deg",
        type=float,
        default=90.0,
        help=(
            "Wrist roll in degrees. Used in Cartesian grasp as local hand roll. "
            "Try -90 if the wrist is flipped."
        ),
    )

    parser.add_argument(
        "-t",
        "--force-top-down-grasp",
        action="store_true",
        help="Legacy shortcut: force top-down grasp orientation for manipulation API grasp.",
    )

    parser.add_argument(
        "-f",
        "--force-horizontal-grasp",
        action="store_true",
        help="Legacy shortcut: force horizontal grasp orientation for manipulation API grasp.",
    )

    parser.add_argument(
        "-r",
        "--force-45-angle-grasp",
        action="store_true",
        help="Legacy shortcut: force 45-degree grasp orientation for manipulation API grasp.",
    )

    parser.add_argument(
        "-s",
        "--force-squeeze-grasp",
        action="store_true",
        help="Legacy shortcut: force squeeze grasp for manipulation API grasp.",
    )

    parser.add_argument(
        "--grasp-constraint-tolerance-rad",
        type=float,
        default=0.35,
        help="Manipulation API grasp orientation tolerance in radians.",
    )

    parser.add_argument(
        "--grasp-timeout-sec",
        type=float,
        default=30.0,
        help="Maximum time to wait for manipulation API feedback.",
    )

    parser.add_argument(
        "--hold-sec",
        type=float,
        default=3.0,
        help="Seconds to hold object after grasp attempt before releasing.",
    )

    parser.set_defaults(return_to_start=True)

    parser.add_argument(
        "--return-to-start",
        dest="return_to_start",
        action="store_true",
        help="Walk back to starting pose after grasp. Enabled by default.",
    )

    parser.add_argument(
        "--no-return-to-start",
        dest="return_to_start",
        action="store_false",
        help="Do not walk back to starting pose after grasp.",
    )

    parser.add_argument(
        "--return-timeout-sec",
        type=float,
        default=15.0,
        help="Timeout for return-to-start command.",
    )

    parser.add_argument(
        "--power-off",
        action="store_true",
        help="Power off robot at the end.",
    )

    return parser.parse_args()


def main():
    options = parse_arguments()

    grasp_constraint_count = sum(
        (
            options.force_top_down_grasp,
            options.force_horizontal_grasp,
            options.force_45_angle_grasp,
            options.force_squeeze_grasp,
        )
    )

    if grasp_constraint_count > 1:
        print("Error: choose at most one legacy force-* grasp constraint.")
        return False

    try:
        run(options)
        return True
    except Exception:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Script failed.")
        return False
    finally:
        try:
            if "options" in locals() and getattr(options, "show_image", False):
                cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    if not main():
        sys.exit(1)
