#!/usr/bin/env python3
# Copyright (c) 2026 Boston Dynamics, Inc. All rights reserved.
#
# AprilTag-only multi-camera Spot grasp script.
#
# Features:
#   - Scans all Spot visual cameras for AprilTags.
#   - Uses body-camera detections to rotate Spot toward the target.
#   - Uses a vertical hand-camera sweep to handle target height / limited hand-camera FOV.
#   - Re-acquires target in hand_color_image.
#   - Uses Spot Manipulation API PickObjectInImage for grasping.
#
# Recommended first run:
#   python3 auto_apriltag_multicam_grasp.py ROBOT_IP --dry-run --show-image
#
# Grasp run:
#   python3 auto_apriltag_multicam_grasp.py ROBOT_IP --show-image
#
# If Spot rotates the wrong direction:
#   python3 auto_apriltag_multicam_grasp.py ROBOT_IP --invert-yaw

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_a_tform_b, get_vision_tform_body
from bosdyn.client.image import ImageClient, pixel_format_to_numpy_type
from bosdyn.client.lease import ResourceAlreadyClaimedError
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    blocking_stand,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient


HAND_COLOR_SOURCE = "hand_color_image"
HAND_DEPTH_SOURCE = "hand_depth_in_hand_color_frame"

DEFAULT_VISUAL_CAMERAS = [
    HAND_COLOR_SOURCE,
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
]

# Approximate yaw direction of each camera relative to Spot body frame.
# Positive yaw means rotate left/counter-clockwise in Spot body frame.
CAMERA_YAW_HINT_RAD = {
    HAND_COLOR_SOURCE: 0.0,
    "frontleft_fisheye_image": 0.35,
    "frontright_fisheye_image": -0.35,
    "left_fisheye_image": 1.35,
    "right_fisheye_image": -1.35,
    "back_fisheye_image": math.pi,
}


@dataclass
class TagDetection:
    source_name: str
    center_xy: Tuple[int, int]
    bbox_xywh: Tuple[int, int, int, int]
    area_px: float
    image_response: object
    image_array: np.ndarray


def verify_estop(robot):
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        raise RuntimeError("Robot is estopped. Clear the estop before running this script.")


def safe_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in text)


def output_dir(config) -> Path:
    path = Path(config.output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(config, image: np.ndarray, filename: str) -> Path:
    path = output_dir(config) / filename
    cv2.imwrite(str(path), image)
    return path


def image_proto_to_array(image_proto) -> np.ndarray:
    """Convert Spot ImageResponse image data to a numpy array."""
    img = image_proto.shot.image

    if img.format == image_pb2.Image.FORMAT_RAW:
        dtype = pixel_format_to_numpy_type(img.pixel_format)
        arr = np.frombuffer(img.data, dtype=dtype)

        rows = img.rows
        cols = img.cols
        expected_single_channel = rows * cols

        if arr.size == expected_single_channel:
            return arr.reshape(rows, cols)

        if expected_single_channel > 0 and arr.size % expected_single_channel == 0:
            channels = arr.size // expected_single_channel
            return arr.reshape(rows, cols, channels)

        raise RuntimeError(
            f"Could not reshape RAW image from source {image_proto.source.name}: "
            f"rows={rows}, cols={cols}, data_size={arr.size}"
        )

    decoded = cv2.imdecode(np.frombuffer(img.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError(f"Could not decode image from source {image_proto.source.name}.")
    return decoded


def displayable_image(image: np.ndarray) -> np.ndarray:
    """Return BGR display image for drawing text/circles."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image.copy()


def show_debug(config, image: np.ndarray, title: str = "Spot AprilTag debug"):
    if not config.show_image:
        return

    cv2.imshow(title, image)
    cv2.waitKey(1)


def get_marker_dictionary_id(dict_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Install opencv-contrib-python, not only opencv-python."
        )

    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"OpenCV marker dictionary not available: {dict_name}")

    return getattr(cv2.aruco, dict_name)


def detect_apriltag_in_array(
    image: np.ndarray,
    dict_name: str,
    min_area_px: float,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int, int, int], float]]:
    """Return center, bbox, area for largest detected AprilTag."""
    if image is None:
        return None

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return None

    try:
        dict_id = get_marker_dictionary_id(dict_name)
    except ValueError:
        return None

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if corners is None or ids is None or len(corners) == 0:
        return None

    best = None

    for corner_set in corners:
        pts = corner_set.reshape((4, 2)).astype(np.float32)
        area = float(cv2.contourArea(pts.astype(np.int32)))

        if area < min_area_px:
            continue

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        bbox = cv2.boundingRect(pts.astype(np.int32))

        if best is None or area > best[2]:
            best = ((cx, cy), bbox, area)

    return best


def draw_detection(image: np.ndarray, detection: TagDetection, label: str) -> np.ndarray:
    disp = displayable_image(image)
    x, y = detection.center_xy
    bx, by, bw, bh = detection.bbox_xywh

    cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
    cv2.circle(disp, (x, y), 10, (0, 255, 0), 2)
    cv2.putText(
        disp,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )
    return disp


def list_available_image_sources(image_client: ImageClient, logger) -> List[str]:
    sources = image_client.list_image_sources()
    names = []

    for src in sources:
        if hasattr(src, "name"):
            names.append(src.name)
        else:
            names.append(str(src))

    logger.info("Available image sources: %s", ", ".join(names))
    return names


def filter_available_sources(requested: Sequence[str], available: Sequence[str], logger) -> List[str]:
    available_set = set(available)
    filtered = [src for src in requested if src in available_set]

    missing = [src for src in requested if src not in available_set]
    if missing:
        logger.warning("Skipping unavailable image sources: %s", ", ".join(missing))

    if not filtered:
        raise RuntimeError("None of the requested visual camera sources are available.")

    return filtered


def scan_sources_for_tag(
    image_client: ImageClient,
    source_names: Sequence[str],
    config,
    logger,
) -> Optional[TagDetection]:
    """Scan a list of image sources once and return the largest AprilTag detection."""
    responses = image_client.get_image_from_sources(list(source_names))

    best_detection = None

    for response in responses:
        source_name = response.source.name

        try:
            arr = image_proto_to_array(response)
        except Exception as exc:
            logger.warning("Failed to decode image from %s: %s", source_name, exc)
            continue

        detected = detect_apriltag_in_array(
            arr,
            dict_name=config.apriltag_dict,
            min_area_px=config.min_tag_area,
        )

        if detected is None:
            continue

        center, bbox, area = detected
        det = TagDetection(
            source_name=source_name,
            center_xy=center,
            bbox_xywh=bbox,
            area_px=area,
            image_response=response,
            image_array=arr,
        )

        if best_detection is None or det.area_px > best_detection.area_px:
            best_detection = det

    if best_detection is not None:
        logger.info(
            "AprilTag found in %s at %s, area=%.1f px",
            best_detection.source_name,
            best_detection.center_xy,
            best_detection.area_px,
        )

        annotated = draw_detection(
            best_detection.image_array,
            best_detection,
            f"Tag in {best_detection.source_name}",
        )
        save_image(
            config,
            annotated,
            f"tag_found_{safe_filename(best_detection.source_name)}.jpg",
        )
        show_debug(config, annotated)

    return best_detection


def capture_hand_tag(
    image_client: ImageClient,
    config,
    logger,
) -> Tuple[Optional[TagDetection], Optional[object], Optional[object]]:
    """Capture hand color image and optional aligned hand depth image."""
    sources = [config.hand_source]

    if config.depth_source and getattr(config, "depth_available", False):
        sources.append(config.depth_source)

    try:
        responses = image_client.get_image_from_sources(sources)
    except Exception as exc:
        logger.warning("Could not capture hand image/depth: %s", exc)
        return None, None, None

    by_name = {resp.source.name: resp for resp in responses}
    color_response = by_name.get(config.hand_source)
    depth_response = by_name.get(config.depth_source)

    if color_response is None:
        logger.warning("Hand color response not returned.")
        return None, None, depth_response

    arr = image_proto_to_array(color_response)

    detected = detect_apriltag_in_array(
        arr,
        dict_name=config.apriltag_dict,
        min_area_px=config.min_tag_area,
    )

    if detected is None:
        return None, color_response, depth_response

    center, bbox, area = detected

    det = TagDetection(
        source_name=color_response.source.name,
        center_xy=center,
        bbox_xywh=bbox,
        area_px=area,
        image_response=color_response,
        image_array=arr,
    )

    logger.info("AprilTag visible in hand camera at %s, area=%.1f px", center, area)

    annotated = draw_detection(arr, det, "Tag in hand camera")
    save_image(config, annotated, "tag_found_hand_camera.jpg")
    show_debug(config, annotated)

    return det, color_response, depth_response


def get_depth_near_pixel(
    depth_response,
    x: int,
    y: int,
    radius: int = 4,
) -> Optional[float]:
    if depth_response is None:
        return None

    img = depth_response.shot.image

    if img.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        return None

    depth = image_proto_to_array(depth_response)

    if depth.ndim != 2:
        return None

    h, w = depth.shape

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    crop = depth[y0:y1, x0:x1]
    valid = crop[crop > 0]

    if valid.size == 0:
        return None

    depth_mm = float(np.median(valid))
    return depth_mm / 1000.0


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def stop_body(command_client: RobotCommandClient):
    cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.0, v_y=0.0, v_rot=0.0)
    command_client.robot_command(cmd, end_time_secs=time.time() + 0.25)


def rotate_body_towards_detection(
    command_client: RobotCommandClient,
    detection: TagDetection,
    config,
    logger,
):
    source = detection.source_name
    source_yaw = CAMERA_YAW_HINT_RAD.get(source, 0.0)

    width = detection.image_array.shape[1]
    target_x = detection.center_xy[0]

    # Positive when tag is left of image center.
    pixel_error_norm = ((width / 2.0) - float(target_x)) / (width / 2.0)
    pixel_yaw = pixel_error_norm * config.pixel_yaw_gain

    yaw_step = normalize_angle(source_yaw + pixel_yaw)

    if config.invert_yaw:
        yaw_step = -yaw_step

    yaw_step = float(np.clip(yaw_step, -config.max_turn_step, config.max_turn_step))

    if abs(yaw_step) < config.min_turn_step:
        logger.info("Yaw correction %.3f rad is below threshold; not rotating.", yaw_step)
        return

    yaw_rate = math.copysign(config.yaw_rate, yaw_step)
    duration = min(max(abs(yaw_step) / config.yaw_rate, 0.35), config.max_turn_duration)

    logger.info(
        "Rotating body toward %s: yaw_step=%.3f rad, yaw_rate=%.3f rad/s, duration=%.2f s",
        source,
        yaw_step,
        yaw_rate,
        duration,
    )

    cmd = RobotCommandBuilder.synchro_velocity_command(
        v_x=0.0,
        v_y=0.0,
        v_rot=yaw_rate,
    )
    command_client.robot_command(cmd, end_time_secs=time.time() + duration)
    time.sleep(duration + config.settle_time)
    stop_body(command_client)
    time.sleep(config.settle_time)


def parse_sweep_offsets(offsets: str) -> List[float]:
    return [float(item.strip()) for item in offsets.split(",") if item.strip()]


def vertical_hand_sweep_for_tag(
    command_client: RobotCommandClient,
    robot_state_client: RobotStateClient,
    image_client: ImageClient,
    config,
    logger,
) -> Tuple[Optional[TagDetection], Optional[object], Optional[object]]:
    """Move the hand camera vertically while checking for the tag."""
    offsets = parse_sweep_offsets(config.arm_sweep_offsets)

    robot_state = robot_state_client.get_robot_state()
    vision_tform_hand = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        VISION_FRAME_NAME,
        "hand",
    )

    base_x = vision_tform_hand.x
    base_y = vision_tform_hand.y
    base_z = vision_tform_hand.z
    base_rot = vision_tform_hand.rot

    logger.info("Starting vertical hand-camera sweep with offsets: %s", offsets)

    for idx, dz in enumerate(offsets):
        target_z = base_z + dz

        logger.info(
            "Hand sweep %d/%d: moving hand camera to z offset %.3f m",
            idx + 1,
            len(offsets),
            dz,
        )

        cmd = RobotCommandBuilder.arm_pose_command(
            base_x,
            base_y,
            target_z,
            base_rot.w,
            base_rot.x,
            base_rot.y,
            base_rot.z,
            VISION_FRAME_NAME,
            config.arm_move_seconds,
        )

        try:
            cmd_id = command_client.robot_command(cmd)
            arrived = block_until_arm_arrives(
                command_client,
                cmd_id,
                timeout_sec=config.arm_move_seconds + 2.0,
            )
            if not arrived:
                logger.warning("Arm did not fully reach sweep pose; continuing.")
        except Exception as exc:
            logger.warning("Arm sweep pose failed: %s", exc)
            continue

        time.sleep(config.settle_time)

        det, color_response, depth_response = capture_hand_tag(image_client, config, logger)
        if det is not None:
            logger.info("Tag found during vertical hand sweep.")
            return det, color_response, depth_response

    logger.info("Vertical hand sweep did not find the tag.")
    return None, None, None


def acquire_tag_in_hand_camera(
    command_client: RobotCommandClient,
    robot_state_client: RobotStateClient,
    image_client: ImageClient,
    visual_sources: Sequence[str],
    config,
    logger,
) -> Tuple[TagDetection, object, Optional[object]]:
    """
    Main search logic:
      1. Try hand camera.
      2. Scan all cameras.
      3. If body camera sees tag, rotate toward it.
      4. Sweep hand camera vertically.
      5. Repeat.
    """
    for attempt in range(1, config.max_search_attempts + 1):
        logger.info("Search attempt %d/%d", attempt, config.max_search_attempts)

        hand_det, hand_color, hand_depth = capture_hand_tag(image_client, config, logger)
        if hand_det is not None:
            return hand_det, hand_color, hand_depth

        all_cam_det = scan_sources_for_tag(image_client, visual_sources, config, logger)

        if all_cam_det is not None:
            if all_cam_det.source_name != config.hand_source:
                rotate_body_towards_detection(command_client, all_cam_det, config, logger)

                hand_det, hand_color, hand_depth = capture_hand_tag(image_client, config, logger)
                if hand_det is not None:
                    return hand_det, hand_color, hand_depth

            sweep_det, sweep_color, sweep_depth = vertical_hand_sweep_for_tag(
                command_client,
                robot_state_client,
                image_client,
                config,
                logger,
            )

            if sweep_det is not None:
                return sweep_det, sweep_color, sweep_depth

        else:
            logger.info("No AprilTag found in any visual camera. Trying vertical hand sweep.")
            sweep_det, sweep_color, sweep_depth = vertical_hand_sweep_for_tag(
                command_client,
                robot_state_client,
                image_client,
                config,
                logger,
            )

            if sweep_det is not None:
                return sweep_det, sweep_color, sweep_depth

        time.sleep(config.settle_time)

    raise RuntimeError("Failed to acquire AprilTag in the hand camera.")


WALK_GAZE_MODE_MAP = {
    "AUTO_GAZE": manipulation_api_pb2.PICK_AUTO_GAZE,
    "AUTO_WALK_AND_GAZE": manipulation_api_pb2.PICK_AUTO_WALK_AND_GAZE,
    "NO_AUTO_WALK_OR_GAZE": manipulation_api_pb2.PICK_NO_AUTO_WALK_OR_GAZE,
    "PLAN_ONLY": manipulation_api_pb2.PICK_PLAN_ONLY,
}


def add_grasp_constraint(config, grasp, robot_state_client):
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=-1)
        else:
            axis_on_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=1)

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper
        )
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align
        )
        constraint.vector_alignment_with_tolerance.threshold_radians = config.grasp_constraint_tolerance

    elif config.force_45_angle_grasp:
        robot_state = robot_state_client.get_robot_state()
        vision_tform_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
        body_q_grasp = bosdyn.client.frame_helpers.math_helpers.Quat.from_pitch(0.785398)
        vision_q_grasp = vision_tform_body.rotation * body_q_grasp

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_q_grasp.to_proto())
        constraint.rotation_with_tolerance.threshold_radians = config.grasp_constraint_tolerance

    elif config.force_squeeze_grasp:
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def perform_grasp(
    manipulation_client: ManipulationApiClient,
    robot_state_client: RobotStateClient,
    image_client: ImageClient,
    config,
    logger,
) -> bool:
    """
    Re-acquire from hand camera immediately before grasping, then send PickObjectInImage.
    """
    logger.info("Final re-acquisition from hand camera before grasp.")

    final_det = None
    final_color_response = None
    final_depth_response = None

    for attempt in range(1, config.final_reacquire_attempts + 1):
        det, color_response, depth_response = capture_hand_tag(image_client, config, logger)
        if det is not None:
            final_det = det
            final_color_response = color_response
            final_depth_response = depth_response
            break

        logger.info("Final re-acquire attempt %d failed.", attempt)
        time.sleep(config.settle_time)

    if final_det is None or final_color_response is None:
        raise RuntimeError("Could not re-acquire AprilTag in hand camera before grasp.")

    target_x = int(final_det.center_xy[0] + config.pixel_offset_x)
    target_y = int(final_det.center_xy[1] + config.pixel_offset_y)

    img_h, img_w = final_det.image_array.shape[:2]
    target_x = int(np.clip(target_x, 0, img_w - 1))
    target_y = int(np.clip(target_y, 0, img_h - 1))

    depth_m = get_depth_near_pixel(
        final_depth_response,
        target_x,
        target_y,
        radius=config.depth_sample_radius,
    )

    if depth_m is not None:
        logger.info("Depth near final target pixel: %.3f m", depth_m)
    else:
        logger.info("No valid aligned hand depth at target pixel; manipulation API may still plan.")

    annotated = draw_detection(
        final_det.image_array,
        final_det,
        f"Final grasp pixel: ({target_x}, {target_y})",
    )
    cv2.circle(annotated, (target_x, target_y), 5, (0, 0, 255), -1)
    save_image(config, annotated, "final_grasp_target.jpg")
    show_debug(config, annotated)

    if config.dry_run:
        logger.info("Dry-run enabled. Not sending manipulation grasp command.")
        return False

    if not final_color_response.source.HasField("pinhole"):
        raise RuntimeError(
            f"Image source {final_color_response.source.name} has no pinhole camera model. "
            "Use hand_color_image for grasping."
        )

    target_pixel = geometry_pb2.Vec2(x=target_x, y=target_y)

    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=target_pixel,
        transforms_snapshot_for_camera=final_color_response.shot.transforms_snapshot,
        frame_name_image_sensor=final_color_response.shot.frame_name_image_sensor,
        camera_model=final_color_response.source.pinhole,
    )

    grasp.walk_gaze_mode = WALK_GAZE_MODE_MAP.get(
        config.walk_gaze_mode,
        manipulation_api_pb2.PICK_AUTO_GAZE,
    )

    add_grasp_constraint(config, grasp, robot_state_client)

    logger.info(
        "Sending manipulation grasp command at pixel (%d, %d), walk_gaze_mode=%s",
        target_x,
        target_y,
        config.walk_gaze_mode,
    )

    request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
    cmd_response = manipulation_client.manipulation_api_command(
        manipulation_api_request=request
    )

    start_time = time.time()
    last_state = None

    while time.time() - start_time < config.grasp_timeout:
        time.sleep(0.25)

        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id
        )
        response = manipulation_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request
        )

        state = response.current_state

        if state != last_state:
            try:
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(state)
            except Exception:
                state_name = str(state)

            logger.info("Manipulation state: %s", state_name)
            last_state = state

        if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
            logger.info("Grasp succeeded.")
            return True

        if state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
            logger.warning("Grasp failed.")
            return False

    logger.warning("Grasp feedback timed out after %.1f seconds.", config.grasp_timeout)
    return False


def run(config):
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk("AprilTagMultiCamGraspClient")
    robot = sdk.create_robot(config.hostname)

    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this script."
    verify_estop(robot)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    try:
        lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
            lease_client,
            must_acquire=True,
            return_at_exit=True,
        )
    except ResourceAlreadyClaimedError:
        if not config.take_lease:
            raise

        robot.logger.warning("Body lease already claimed. Taking lease because --take-lease was set.")
        lease_client.take(resource="body")
        lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
            lease_client,
            must_acquire=False,
            return_at_exit=True,
        )

    start_x = None
    start_y = None
    start_yaw = None

    with lease_keep_alive:
        robot.logger.info("Powering on robot.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power-on failed."

        robot.logger.info("Standing.")
        blocking_stand(command_client, timeout_sec=10)

        initial_state = robot_state_client.get_robot_state()
        vision_tform_body = get_vision_tform_body(initial_state.kinematic_state.transforms_snapshot)

        start_x = vision_tform_body.x
        start_y = vision_tform_body.y
        start_yaw = vision_tform_body.rot.to_yaw()

        robot.logger.info(
            "Recorded start pose in vision frame: x=%.3f, y=%.3f, yaw=%.3f",
            start_x,
            start_y,
            start_yaw,
        )

        available_sources = list_available_image_sources(image_client, robot.logger)

        visual_sources = filter_available_sources(
            config.visual_sources,
            available_sources,
            robot.logger,
        )

        config.depth_available = config.depth_source in available_sources

        if not config.depth_available:
            robot.logger.warning(
                "Depth source %s is not available. Grasp may still work, but local depth logging is disabled.",
                config.depth_source,
            )

        robot.logger.info("Moving arm to ready position.")
        command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        time.sleep(2.0)

        robot.logger.info("Opening gripper.")
        command_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0))
        time.sleep(1.0)

        hand_det, hand_color_response, hand_depth_response = acquire_tag_in_hand_camera(
            command_client,
            robot_state_client,
            image_client,
            visual_sources,
            config,
            robot.logger,
        )

        robot.logger.info(
            "Target acquired in hand camera at %s, area=%.1f px",
            hand_det.center_xy,
            hand_det.area_px,
        )

        grasp_success = perform_grasp(
            manipulation_client,
            robot_state_client,
            image_client,
            config,
            robot.logger,
        )

        if not config.dry_run:
            robot.logger.info("Holding for %.1f seconds.", config.hold_seconds)
            time.sleep(config.hold_seconds)

            if not config.keep_holding:
                robot.logger.info("Opening gripper to release.")
                command_client.robot_command(
                    RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
                )
                time.sleep(1.5)

            if not config.no_stow:
                robot.logger.info("Stowing arm.")
                command_client.robot_command(RobotCommandBuilder.arm_stow_command())
                time.sleep(3.0)

        if config.return_to_start and start_x is not None:
            robot.logger.info("Returning to recorded start pose.")
            return_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=start_x,
                goal_y=start_y,
                goal_heading=start_yaw,
                frame_name=VISION_FRAME_NAME,
            )
            command_client.robot_command(return_cmd, end_time_secs=time.time() + 15.0)
            time.sleep(10.0)

        if config.power_off_at_end:
            robot.logger.info("Powering off robot.")
            robot.power_off(cut_immediately=False, timeout_sec=20)

        if config.show_image:
            cv2.destroyAllWindows()

        return grasp_success or config.dry_run


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AprilTag-only multi-camera Spot grasp script."
    )

    bosdyn.client.util.add_base_arguments(parser)

    parser.add_argument(
        "--hand-source",
        default=HAND_COLOR_SOURCE,
        help="Hand camera source used for final grasping.",
    )
    parser.add_argument(
        "--depth-source",
        default=HAND_DEPTH_SOURCE,
        help="Aligned hand depth source used for depth logging.",
    )
    parser.add_argument(
        "--visual-sources",
        nargs="+",
        default=DEFAULT_VISUAL_CAMERAS,
        help="Visual image sources to scan for AprilTags.",
    )
    parser.add_argument(
        "--apriltag-dict",
        default="DICT_APRILTAG_36H11",
        help="OpenCV ArUco/AprilTag dictionary name.",
    )
    parser.add_argument(
        "--min-tag-area",
        type=float,
        default=80.0,
        help="Minimum AprilTag contour area in pixels.",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Show debug images live using OpenCV.",
    )
    parser.add_argument(
        "--output-dir",
        default="spot_apriltag_grasp_output",
        help="Folder for debug snapshots.",
    )

    parser.add_argument(
        "--max-search-attempts",
        type=int,
        default=8,
        help="Maximum search/align/sweep attempts before failing.",
    )
    parser.add_argument(
        "--arm-sweep-offsets",
        default="0,-0.10,0.10,-0.20,0.20,-0.30,0.30",
        help="Comma-separated vertical hand sweep offsets in meters.",
    )
    parser.add_argument(
        "--arm-move-seconds",
        type=float,
        default=1.5,
        help="Requested duration for each arm sweep move.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.35,
        help="Small wait after arm/body movement before capturing images.",
    )

    parser.add_argument(
        "--yaw-rate",
        type=float,
        default=0.35,
        help="Body yaw rate in rad/s while rotating toward camera detections.",
    )
    parser.add_argument(
        "--max-turn-step",
        type=float,
        default=0.65,
        help="Maximum body yaw correction per search attempt in radians.",
    )
    parser.add_argument(
        "--min-turn-step",
        type=float,
        default=0.05,
        help="Minimum yaw correction before a body rotation is sent.",
    )
    parser.add_argument(
        "--max-turn-duration",
        type=float,
        default=2.2,
        help="Maximum duration for one body rotation command.",
    )
    parser.add_argument(
        "--pixel-yaw-gain",
        type=float,
        default=0.35,
        help="Extra yaw correction from horizontal pixel offset.",
    )
    parser.add_argument(
        "--invert-yaw",
        action="store_true",
        help="Use this if Spot rotates away from the target instead of toward it.",
    )

    parser.add_argument(
        "--walk-gaze-mode",
        default="AUTO_GAZE",
        choices=list(WALK_GAZE_MODE_MAP.keys()),
        help="Manipulation API walk/gaze behavior.",
    )
    parser.add_argument(
        "--final-reacquire-attempts",
        type=int,
        default=4,
        help="Attempts to re-acquire AprilTag in hand camera before grasp.",
    )
    parser.add_argument(
        "--pixel-offset-x",
        type=int,
        default=0,
        help="Optional final grasp pixel X offset from tag center.",
    )
    parser.add_argument(
        "--pixel-offset-y",
        type=int,
        default=0,
        help="Optional final grasp pixel Y offset from tag center.",
    )
    parser.add_argument(
        "--depth-sample-radius",
        type=int,
        default=4,
        help="Radius around target pixel used for median depth logging.",
    )
    parser.add_argument(
        "--grasp-timeout",
        type=float,
        default=25.0,
        help="Timeout for manipulation feedback.",
    )

    parser.add_argument("--take-lease", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Detect and align, but do not grasp.")
    parser.add_argument("--keep-holding", action="store_true", help="Do not release after grasp.")
    parser.add_argument("--no-stow", action="store_true", help="Do not stow arm at the end.")
    parser.add_argument("--return-to-start", action="store_true", help="Walk back to the recorded start pose.")
    parser.add_argument("--power-off-at-end", action="store_true", help="Power off after the run.")

    parser.add_argument("--force-top-down-grasp", action="store_true")
    parser.add_argument("--force-horizontal-grasp", action="store_true")
    parser.add_argument("--force-45-angle-grasp", action="store_true")
    parser.add_argument("--force-squeeze-grasp", action="store_true")
    parser.add_argument(
        "--grasp-constraint-tolerance",
        type=float,
        default=0.17,
        help="Tolerance in radians for forced grasp orientation constraints.",
    )

    return parser.parse_args()


def main():
    options = parse_arguments()

    num_constraints = sum(
        (
            options.force_top_down_grasp,
            options.force_horizontal_grasp,
            options.force_45_angle_grasp,
            options.force_squeeze_grasp,
        )
    )

    if num_constraints > 1:
        print("Error: choose at most one grasp constraint.")
        return False

    try:
        return run(options)
    except Exception:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Script failed.")
        return False
    finally:
        if getattr(options, "show_image", False):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if not main():
        sys.exit(1)
