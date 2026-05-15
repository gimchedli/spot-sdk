#!/usr/bin/env python3
# Copyright (c) 2026 Boston Dynamics, Inc. All rights reserved.
#
# AprilTag-only multi-camera search + hand-camera grasp for Spot.
#
# Features:
#   - Searches all visual cameras for AprilTags.
#   - Uses body cameras only for coarse yaw alignment.
#   - Uses hand_color_image for the final grasp command.
#   - Keeps live camera view with C to cycle cameras.
#   - Q / E / ESC sends a best-effort stop command and aborts the script.
#   - Supports configurable grasp pixel offsets from AprilTag center.
#
# NOTE:
#   Q / E / ESC is an operator abort, not a certified estop.
#   Keep Spot's real estop available.

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
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient, pixel_format_to_numpy_type
from bosdyn.client.lease import ResourceAlreadyClaimedError
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient


HAND_SOURCE = "hand_color_image"

DEFAULT_VISUAL_SOURCES = [
    HAND_SOURCE,
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
]

# Approximate yaw direction for each camera relative to Spot's body.
# Positive yaw means rotate Spot left.
CAMERA_YAW_HINT_RAD = {
    HAND_SOURCE: 0.0,
    "frontleft_fisheye_image": 0.35,
    "frontright_fisheye_image": -0.35,
    "left_fisheye_image": 1.35,
    "right_fisheye_image": -1.35,
    "back_fisheye_image": math.pi,
}

BODY_CAMERA_PRIORITY = {
    HAND_SOURCE: 0,
    "frontleft_fisheye_image": 1,
    "frontright_fisheye_image": 1,
    "left_fisheye_image": 2,
    "right_fisheye_image": 2,
    "back_fisheye_image": 3,
}

WALK_GAZE_MODE_MAP = {
    "AUTO_GAZE": manipulation_api_pb2.PICK_AUTO_GAZE,
    "AUTO_WALK_AND_GAZE": manipulation_api_pb2.PICK_AUTO_WALK_AND_GAZE,
    "NO_AUTO_WALK_OR_GAZE": manipulation_api_pb2.PICK_NO_AUTO_WALK_OR_GAZE,
    "PLAN_ONLY": manipulation_api_pb2.PICK_PLAN_ONLY,
}


class OperatorAbort(Exception):
    """Raised when the operator aborts from the live camera window."""


@dataclass
class AprilTagDetection:
    source_name: str
    center_xy: Tuple[int, int]
    corners_xy: np.ndarray
    bbox_xywh: Tuple[int, int, int, int]
    area_px: float
    tag_id: int
    image_response: object
    image_array: np.ndarray


def verify_estop(robot):
    estop_client = robot.ensure_client(EstopClient.default_service_name)
    status = estop_client.get_status()

    if status.stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        raise RuntimeError("Robot is estopped. Clear estop before running.")


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)


def get_output_dir(config) -> Path:
    path = Path(config.output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_debug_image(config, image: np.ndarray, filename: str):
    path = get_output_dir(config) / filename
    cv2.imwrite(str(path), image)
    return path


def image_proto_to_array(image_proto) -> np.ndarray:
    image = image_proto.shot.image

    if image.format == image_pb2.Image.FORMAT_RAW:
        dtype = pixel_format_to_numpy_type(image.pixel_format)
        arr = np.frombuffer(image.data, dtype=dtype)

        rows = image.rows
        cols = image.cols
        expected = rows * cols

        if arr.size == expected:
            return arr.reshape(rows, cols)

        if expected > 0 and arr.size % expected == 0:
            channels = arr.size // expected
            return arr.reshape(rows, cols, channels)

        raise RuntimeError(
            f"Could not reshape RAW image from {image_proto.source.name}: "
            f"rows={rows}, cols={cols}, size={arr.size}"
        )

    decoded = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if decoded is None:
        raise RuntimeError(f"Could not decode image from {image_proto.source.name}")

    return decoded


def to_display_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image.copy()


def get_marker_dictionary_id(dict_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is not available. Install opencv-contrib-python.")

    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"OpenCV marker dictionary not available: {dict_name}")

    return getattr(cv2.aruco, dict_name)


def detect_apriltag(
    image: np.ndarray,
    image_response,
    dict_name: str,
    min_area_px: float,
) -> Optional[AprilTagDetection]:
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

    dict_id = get_marker_dictionary_id(dict_name)
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

    for i, corner_set in enumerate(corners):
        pts = corner_set.reshape((4, 2)).astype(np.float32)
        area = float(cv2.contourArea(pts.astype(np.int32)))

        if area < min_area_px:
            continue

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        bbox = cv2.boundingRect(pts.astype(np.int32))
        tag_id = int(ids[i][0])

        detection = AprilTagDetection(
            source_name=image_response.source.name,
            center_xy=(cx, cy),
            corners_xy=pts,
            bbox_xywh=bbox,
            area_px=area,
            tag_id=tag_id,
            image_response=image_response,
            image_array=image,
        )

        if best is None or detection.area_px > best.area_px:
            best = detection

    return best


def compute_grasp_pixel(detection: AprilTagDetection, config) -> Tuple[int, int]:
    """
    Compute final grasp pixel from AprilTag center.

    Offset options:
      --grasp-offset-px-x / --grasp-offset-px-y
        Raw image pixel offsets.

      --grasp-offset-tag-x / --grasp-offset-tag-y
        Offsets along the AprilTag's local image axes.
        Example: --grasp-offset-tag-y 0.5 means half a tag-height below the tag center.
    """
    point = np.array(detection.center_xy, dtype=np.float32)

    # OpenCV ArUco/AprilTag corner order is normally:
    # 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left.
    tl, tr, br, bl = detection.corners_xy

    tag_x_axis = ((tr + br) / 2.0) - ((tl + bl) / 2.0)
    tag_y_axis = ((bl + br) / 2.0) - ((tl + tr) / 2.0)

    point += config.grasp_offset_tag_x * tag_x_axis
    point += config.grasp_offset_tag_y * tag_y_axis
    point += np.array([config.grasp_offset_px_x, config.grasp_offset_px_y], dtype=np.float32)

    height, width = detection.image_array.shape[:2]
    gx = int(np.clip(round(point[0]), 0, width - 1))
    gy = int(np.clip(round(point[1]), 0, height - 1))

    return gx, gy


def draw_detection(
    detection: AprilTagDetection,
    grasp_xy: Optional[Tuple[int, int]] = None,
    label: str = "",
) -> np.ndarray:
    image = to_display_bgr(detection.image_array)

    x, y = detection.center_xy
    bx, by, bw, bh = detection.bbox_xywh

    cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
    cv2.circle(image, (x, y), 8, (255, 0, 0), -1)

    cv2.putText(
        image,
        "tag center",
        (x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
    )

    for idx, point in enumerate(detection.corners_xy.astype(int)):
        cv2.circle(image, tuple(point), 4, (0, 255, 255), -1)
        cv2.putText(
            image,
            str(idx),
            tuple(point + np.array([5, -5])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    if grasp_xy is not None:
        gx, gy = grasp_xy
        cv2.circle(image, (gx, gy), 10, (0, 0, 255), -1)
        cv2.line(image, (x, y), (gx, gy), (0, 0, 255), 2)

        cv2.putText(
            image,
            "GRASP PIXEL",
            (gx + 12, gy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    text = label or f"{detection.source_name}, id={detection.tag_id}, area={detection.area_px:.0f}"

    cv2.putText(
        image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return image


def list_available_sources(image_client: ImageClient, logger) -> List[str]:
    sources = image_client.list_image_sources()
    names = [source.name for source in sources]

    logger.info("Available image sources:")

    for name in names:
        logger.info("  %s", name)

    return names


def filter_sources(requested: Sequence[str], available: Sequence[str], logger) -> List[str]:
    available_set = set(available)
    result = [source for source in requested if source in available_set]
    missing = [source for source in requested if source not in available_set]

    if missing:
        logger.warning("Skipping unavailable image sources: %s", ", ".join(missing))

    if not result:
        raise RuntimeError("No requested image sources are available.")

    return result


def choose_best_detection(detections: Sequence[AprilTagDetection]) -> Optional[AprilTagDetection]:
    if not detections:
        return None

    return sorted(
        detections,
        key=lambda detection: (
            BODY_CAMERA_PRIORITY.get(detection.source_name, 99),
            -detection.area_px,
        ),
    )[0]


def choose_best_non_hand_detection(detections: Sequence[AprilTagDetection]) -> Optional[AprilTagDetection]:
    non_hand = [det for det in detections if det.source_name != HAND_SOURCE]

    if not non_hand:
        return None

    return sorted(
        non_hand,
        key=lambda detection: (
            BODY_CAMERA_PRIORITY.get(detection.source_name, 99),
            -detection.area_px,
        ),
    )[0]


def choose_best_hand_detection(detections: Sequence[AprilTagDetection]) -> Optional[AprilTagDetection]:
    hand = [det for det in detections if det.source_name == HAND_SOURCE]

    if not hand:
        return None

    return max(hand, key=lambda detection: detection.area_px)


def stop_body(command_client: RobotCommandClient):
    command_client.robot_command(
        RobotCommandBuilder.synchro_velocity_command(
            v_x=0.0,
            v_y=0.0,
            v_rot=0.0,
        ),
        end_time_secs=time.time() + 0.25,
    )


def send_operator_abort(command_client: RobotCommandClient, logger):
    """
    Best-effort operator abort.
    This is not a certified estop. Use Spot's real estop for actual safety.
    """
    logger.warning("OPERATOR ABORT REQUESTED FROM CAMERA WINDOW.")

    try:
        stop_body(command_client)
    except Exception as exc:
        logger.warning("Failed to send zero velocity stop command: %s", exc)

    try:
        command_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0))
    except Exception as exc:
        logger.warning("Failed to open gripper during operator abort: %s", exc)


def poll_operator_abort(config, command_client: RobotCommandClient, logger):
    if not config.show_image:
        return

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q"), ord("e"), ord("E"), 27):
        send_operator_abort(command_client, logger)
        raise OperatorAbort("Operator aborted from live camera window.")


def draw_live_overlay(
    image: np.ndarray,
    source_name: str,
    detections: Sequence[AprilTagDetection],
    active_best: Optional[AprilTagDetection],
) -> np.ndarray:
    frame = to_display_bgr(image)

    cv2.putText(
        frame,
        f"VIEW: {source_name}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "C: cycle camera | Q/E/ESC: abort script",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    for detection in detections:
        if detection.source_name != source_name:
            continue

        x, y = detection.center_xy
        bx, by, bw, bh = detection.bbox_xywh

        color = (0, 255, 0)

        if active_best is not None and detection.source_name == active_best.source_name:
            color = (0, 0, 255)

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.circle(frame, (x, y), 8, color, -1)

        cv2.putText(
            frame,
            f"TAG id={detection.tag_id} area={detection.area_px:.0f}",
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    if active_best is not None:
        cv2.putText(
            frame,
            f"BEST: {active_best.source_name}",
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            frame,
            "NO TAG FOUND",
            (10, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return frame


def update_live_camera_view(
    config,
    responses,
    detections: Sequence[AprilTagDetection],
    best_detection: Optional[AprilTagDetection],
    command_client: RobotCommandClient,
    logger,
):
    if not config.show_image:
        return

    if not hasattr(config, "live_camera_index"):
        config.live_camera_index = 0

    if not responses:
        poll_operator_abort(config, command_client, logger)
        return

    response_names = [response.source.name for response in responses]
    config.live_camera_index %= len(response_names)

    selected_response = responses[config.live_camera_index]
    selected_name = selected_response.source.name

    try:
        image = image_proto_to_array(selected_response)
    except Exception as exc:
        logger.warning("Could not decode live camera image from %s: %s", selected_name, exc)
        poll_operator_abort(config, command_client, logger)
        return

    frame = draw_live_overlay(
        image=image,
        source_name=selected_name,
        detections=detections,
        active_best=best_detection,
    )

    cv2.imshow("Live Spot Camera - AprilTag Search", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("c"), ord("C")):
        config.live_camera_index = (config.live_camera_index + 1) % len(response_names)
    elif key in (ord("q"), ord("Q"), ord("e"), ord("E"), 27):
        send_operator_abort(command_client, logger)
        raise OperatorAbort("Operator aborted from live camera window.")


def scan_all_cameras_once(
    image_client: ImageClient,
    source_names: Sequence[str],
    command_client: RobotCommandClient,
    config,
    logger,
) -> List[AprilTagDetection]:
    detections = []
    responses = image_client.get_image_from_sources(list(source_names))

    for response in responses:
        try:
            image = image_proto_to_array(response)
        except Exception as exc:
            logger.warning("Could not decode %s: %s", response.source.name, exc)
            continue

        try:
            detection = detect_apriltag(
                image=image,
                image_response=response,
                dict_name=config.apriltag_dict,
                min_area_px=config.min_tag_area,
            )
        except Exception as exc:
            logger.warning("AprilTag detection failed on %s: %s", response.source.name, exc)
            continue

        if detection is not None:
            detections.append(detection)

            debug = draw_detection(detection, label=f"FOUND in {detection.source_name}")
            save_debug_image(config, debug, f"detected_{safe_name(detection.source_name)}.jpg")

    best_detection = choose_best_detection(detections)

    update_live_camera_view(
        config=config,
        responses=responses,
        detections=detections,
        best_detection=best_detection,
        command_client=command_client,
        logger=logger,
    )

    return detections


def send_yaw_velocity_for_step(
    command_client: RobotCommandClient,
    yaw_step_rad: float,
    yaw_rate_rad_s: float,
    config,
    logger,
    reason: str,
):
    if config.invert_yaw:
        yaw_step_rad = -yaw_step_rad

    yaw_step_rad = float(np.clip(yaw_step_rad, -config.max_turn_step_rad, config.max_turn_step_rad))

    if abs(yaw_step_rad) < config.min_turn_step_rad:
        logger.info("Yaw correction too small. Not rotating. reason=%s", reason)
        return

    yaw_rate = math.copysign(abs(yaw_rate_rad_s), yaw_step_rad)
    duration = abs(yaw_step_rad) / max(abs(yaw_rate_rad_s), 0.001)
    duration = float(np.clip(duration, config.min_turn_duration_s, config.max_turn_duration_s))

    logger.info(
        "%s: yaw_step=%.3f rad, yaw_rate=%.3f rad/s, duration=%.2f s",
        reason,
        yaw_step_rad,
        yaw_rate,
        duration,
    )

    command_client.robot_command(
        RobotCommandBuilder.synchro_velocity_command(
            v_x=0.0,
            v_y=0.0,
            v_rot=yaw_rate,
        ),
        end_time_secs=time.time() + duration,
    )

    time.sleep(duration + config.settle_time_s)
    stop_body(command_client)
    time.sleep(config.settle_time_s)


def rotate_toward_body_camera_detection(
    command_client: RobotCommandClient,
    detection: AprilTagDetection,
    config,
    logger,
):
    source_name = detection.source_name

    if source_name == HAND_SOURCE:
        return

    source_yaw = CAMERA_YAW_HINT_RAD.get(source_name, 0.0)
    width = detection.image_array.shape[1]
    target_x = detection.center_xy[0]

    # Positive when the tag is on the left side of the image.
    pixel_error_norm = ((width / 2.0) - float(target_x)) / (width / 2.0)
    pixel_yaw = pixel_error_norm * config.pixel_yaw_gain
    yaw_step = source_yaw + pixel_yaw

    while yaw_step > math.pi:
        yaw_step -= 2.0 * math.pi

    while yaw_step < -math.pi:
        yaw_step += 2.0 * math.pi

    send_yaw_velocity_for_step(
        command_client=command_client,
        yaw_step_rad=yaw_step,
        yaw_rate_rad_s=config.yaw_rate_rad_s,
        config=config,
        logger=logger,
        reason=f"Target seen in {source_name}. Rotating toward it",
    )


def center_hand_detection_if_needed(
    command_client: RobotCommandClient,
    detection: AprilTagDetection,
    config,
    logger,
) -> bool:
    """
    Returns True when the hand-camera target is centered enough for final grasp.
    Returns False after sending a correction command, so caller should re-acquire.
    """
    if config.disable_hand_centering:
        return True

    width = detection.image_array.shape[1]
    target_x = detection.center_xy[0]
    error_norm = ((width / 2.0) - float(target_x)) / (width / 2.0)

    logger.info("Hand camera x-centering error: %.3f", error_norm)

    if abs(error_norm) <= config.hand_center_deadband:
        return True

    yaw_step = error_norm * config.hand_center_yaw_gain

    send_yaw_velocity_for_step(
        command_client=command_client,
        yaw_step_rad=yaw_step,
        yaw_rate_rad_s=config.yaw_rate_rad_s,
        config=config,
        logger=logger,
        reason="Centering AprilTag in hand camera",
    )

    return False


def global_search_turn(command_client: RobotCommandClient, config, logger):
    yaw_rate = config.global_search_yaw_rate_rad_s

    if config.invert_yaw:
        yaw_rate = -yaw_rate

    logger.info(
        "No camera sees AprilTag. Performing global search turn: yaw_rate=%.3f rad/s, duration=%.2f s",
        yaw_rate,
        config.global_search_turn_duration_s,
    )

    command_client.robot_command(
        RobotCommandBuilder.synchro_velocity_command(
            v_x=0.0,
            v_y=0.0,
            v_rot=yaw_rate,
        ),
        end_time_secs=time.time() + config.global_search_turn_duration_s,
    )

    time.sleep(config.global_search_turn_duration_s + config.settle_time_s)
    stop_body(command_client)
    time.sleep(config.settle_time_s)


def acquire_centered_hand_detection(
    image_client: ImageClient,
    visual_sources: Sequence[str],
    command_client: RobotCommandClient,
    config,
    logger,
) -> AprilTagDetection:
    """
    Uses all cameras for search/alignment, but returns only a hand camera detection.
    """
    for attempt in range(1, config.max_search_attempts + 1):
        logger.info("Search attempt %d/%d", attempt, config.max_search_attempts)

        detections = scan_all_cameras_once(
            image_client=image_client,
            source_names=visual_sources,
            command_client=command_client,
            config=config,
            logger=logger,
        )

        hand_detection = choose_best_hand_detection(detections)

        if hand_detection is not None:
            logger.info(
                "AprilTag visible in hand camera: center=%s, area=%.1f, id=%d",
                hand_detection.center_xy,
                hand_detection.area_px,
                hand_detection.tag_id,
            )

            if hand_detection.area_px < config.min_final_tag_area:
                logger.info(
                    "Hand-camera tag area %.1f is below final threshold %.1f. Continuing alignment.",
                    hand_detection.area_px,
                    config.min_final_tag_area,
                )
            elif center_hand_detection_if_needed(command_client, hand_detection, config, logger):
                return hand_detection
            else:
                continue

        best_non_hand = choose_best_non_hand_detection(detections)

        if best_non_hand is not None:
            logger.info(
                "Best non-hand detection: source=%s, center=%s, area=%.1f, id=%d",
                best_non_hand.source_name,
                best_non_hand.center_xy,
                best_non_hand.area_px,
                best_non_hand.tag_id,
            )

            rotate_toward_body_camera_detection(command_client, best_non_hand, config, logger)
            continue

        global_search_turn(command_client, config, logger)

    raise RuntimeError(
        f"Could not acquire a centered AprilTag in {HAND_SOURCE} after "
        f"{config.max_search_attempts} attempts."
    )


def add_grasp_constraint(config, grasp, robot_state_client):
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

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

        constraint.vector_alignment_with_tolerance.threshold_radians = (
            config.grasp_constraint_tolerance_rad
        )

    elif config.force_45_angle_grasp:
        robot_state = robot_state_client.get_robot_state()
        vision_tform_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        body_q_grasp = math_helpers.Quat.from_pitch(0.785398)
        vision_q_grasp = vision_tform_body.rotation * body_q_grasp

        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_q_grasp.to_proto())
        constraint.rotation_with_tolerance.threshold_radians = config.grasp_constraint_tolerance_rad

    elif config.force_squeeze_grasp:
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def final_reacquire_from_hand_camera(
    image_client: ImageClient,
    command_client: RobotCommandClient,
    config,
    logger,
) -> AprilTagDetection:
    for attempt in range(1, config.final_reacquire_attempts + 1):
        responses = image_client.get_image_from_sources([HAND_SOURCE])

        detections = []
        detection = None

        if len(responses) == 1:
            image = image_proto_to_array(responses[0])

            detection = detect_apriltag(
                image=image,
                image_response=responses[0],
                dict_name=config.apriltag_dict,
                min_area_px=config.min_tag_area,
            )

            if detection is not None:
                detections.append(detection)

        update_live_camera_view(
            config=config,
            responses=responses,
            detections=detections,
            best_detection=detection,
            command_client=command_client,
            logger=logger,
        )

        if detection is not None and detection.area_px >= config.min_final_tag_area:
            logger.info(
                "Final hand-camera detection: center=%s, area=%.1f, id=%d",
                detection.center_xy,
                detection.area_px,
                detection.tag_id,
            )

            return detection

        logger.info("Final reacquire attempt %d/%d failed.", attempt, config.final_reacquire_attempts)
        time.sleep(config.settle_time_s)

    raise RuntimeError("Could not re-acquire AprilTag in hand camera before grasp.")


def perform_grasp(
    manipulation_client: ManipulationApiClient,
    robot_state_client: RobotStateClient,
    image_client: ImageClient,
    command_client: RobotCommandClient,
    config,
    logger,
) -> bool:
    detection = final_reacquire_from_hand_camera(
        image_client=image_client,
        command_client=command_client,
        config=config,
        logger=logger,
    )

    grasp_x, grasp_y = compute_grasp_pixel(detection, config)

    annotated = draw_detection(
        detection,
        grasp_xy=(grasp_x, grasp_y),
        label=f"Final grasp pixel: ({grasp_x}, {grasp_y})",
    )

    save_debug_image(config, annotated, "final_grasp_pixel.jpg")

    if config.show_image:
        cv2.imshow("Final grasp pixel", annotated)
        poll_operator_abort(config, command_client, logger)

    logger.info("Final tag center: %s", detection.center_xy)
    logger.info("Final grasp pixel: (%d, %d)", grasp_x, grasp_y)

    if config.dry_run:
        logger.info("Dry-run enabled. Not sending grasp command.")
        return True

    target_pixel = geometry_pb2.Vec2(x=grasp_x, y=grasp_y)

    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=target_pixel,
        transforms_snapshot_for_camera=detection.image_response.shot.transforms_snapshot,
        frame_name_image_sensor=detection.image_response.shot.frame_name_image_sensor,
        camera_model=detection.image_response.source.pinhole,
    )

    grasp.walk_gaze_mode = WALK_GAZE_MODE_MAP.get(
        config.walk_gaze_mode,
        manipulation_api_pb2.PICK_NO_AUTO_WALK_OR_GAZE,
    )

    add_grasp_constraint(config, grasp, robot_state_client)

    request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

    logger.info(
        "Sending grasp command using %s at pixel (%d, %d), walk_gaze_mode=%s",
        detection.source_name,
        grasp_x,
        grasp_y,
        config.walk_gaze_mode,
    )

    cmd_response = manipulation_client.manipulation_api_command(
        manipulation_api_request=request
    )

    start_time = time.time()
    last_state = None

    while time.time() - start_time < config.grasp_timeout_s:
        poll_operator_abort(config, command_client, logger)

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

    logger.warning("Grasp timed out after %.1f seconds.", config.grasp_timeout_s)
    return False


def run(config):
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk("AprilTagSearchAndGraspClient")
    robot = sdk.create_robot(config.hostname)

    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm."
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

        robot.logger.warning("Lease already claimed. Taking lease because --take-lease was set.")
        lease_client.take(resource="body")

        lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
            lease_client,
            must_acquire=False,
            return_at_exit=True,
        )

    with lease_keep_alive:
        robot.logger.info("Powering on.")
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
            "Recorded start pose: x=%.3f, y=%.3f, yaw=%.3f",
            start_x,
            start_y,
            start_yaw,
        )

        available_sources = list_available_sources(image_client, robot.logger)
        visual_sources = filter_sources(config.visual_sources, available_sources, robot.logger)

        if HAND_SOURCE not in visual_sources:
            raise RuntimeError(f"{HAND_SOURCE} must be available because final grasp uses the hand camera.")

        robot.logger.info("Using visual sources: %s", ", ".join(visual_sources))

        robot.logger.info("Moving arm to ready pose.")
        command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        time.sleep(2.0)

        robot.logger.info("Opening gripper.")
        command_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0))
        time.sleep(1.0)

        hand_detection = acquire_centered_hand_detection(
            image_client=image_client,
            visual_sources=visual_sources,
            command_client=command_client,
            config=config,
            logger=robot.logger,
        )

        acquired_debug = draw_detection(hand_detection, label="Hand camera acquired target")
        save_debug_image(config, acquired_debug, "hand_camera_acquired.jpg")

        success = perform_grasp(
            manipulation_client=manipulation_client,
            robot_state_client=robot_state_client,
            image_client=image_client,
            command_client=command_client,
            config=config,
            logger=robot.logger,
        )

        if not config.dry_run:
            robot.logger.info("Holding for %.1f seconds.", config.hold_seconds_s)
            time.sleep(config.hold_seconds_s)

            if not config.keep_holding:
                robot.logger.info("Opening gripper.")
                command_client.robot_command(
                    RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
                )
                time.sleep(1.5)

            if not config.no_stow:
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
                end_time_secs=time.time() + config.return_timeout_s,
            )

            time.sleep(min(config.return_timeout_s, 10.0))

        if config.power_off_at_end:
            robot.logger.info("Powering off.")
            robot.power_off(cut_immediately=False, timeout_sec=20)

        if config.show_image:
            cv2.destroyAllWindows()

        return success


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AprilTag-only multi-camera Spot search and hand-camera grasp."
    )

    bosdyn.client.util.add_base_arguments(parser)

    parser.add_argument(
        "--visual-sources",
        nargs="+",
        default=DEFAULT_VISUAL_SOURCES,
        help="Image sources used for AprilTag search.",
    )

    parser.add_argument(
        "--apriltag-dict",
        default="DICT_APRILTAG_36H11",
        help="OpenCV AprilTag dictionary name.",
    )

    parser.add_argument(
        "--min-tag-area",
        type=float,
        default=80.0,
        help="Minimum AprilTag contour area in pixels for any camera.",
    )

    parser.add_argument(
        "--min-final-tag-area",
        type=float,
        default=150.0,
        help="Minimum AprilTag area in hand camera before final grasp.",
    )

    parser.add_argument(
        "--max-search-attempts",
        type=int,
        default=16,
        help="Maximum all-camera search/alignment attempts.",
    )

    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Show live OpenCV camera view. C cycles camera. Q/E/ESC aborts script.",
    )

    parser.add_argument(
        "--output-dir",
        default="spot_apriltag_debug",
        help="Folder where debug images are saved.",
    )

    parser.add_argument(
        "--yaw-rate-rad-s",
        type=float,
        default=0.35,
        help="Yaw rate used when rotating toward detected target.",
    )

    parser.add_argument(
        "--pixel-yaw-gain",
        type=float,
        default=0.35,
        help="Yaw correction from target x-position inside body-camera image.",
    )

    parser.add_argument(
        "--hand-center-yaw-gain",
        type=float,
        default=0.35,
        help="Yaw correction from target x-position inside hand-camera image.",
    )

    parser.add_argument(
        "--hand-center-deadband",
        type=float,
        default=0.18,
        help="Normalized x-error threshold where hand camera is considered centered.",
    )

    parser.add_argument(
        "--disable-hand-centering",
        action="store_true",
        help="Do not rotate to center target inside hand camera before grasp.",
    )

    parser.add_argument(
        "--max-turn-step-rad",
        type=float,
        default=0.70,
        help="Maximum yaw correction per search attempt.",
    )

    parser.add_argument(
        "--min-turn-step-rad",
        type=float,
        default=0.05,
        help="Minimum yaw correction before sending a turn command.",
    )

    parser.add_argument(
        "--min-turn-duration-s",
        type=float,
        default=0.35,
        help="Minimum body turn command duration.",
    )

    parser.add_argument(
        "--max-turn-duration-s",
        type=float,
        default=2.0,
        help="Maximum body turn command duration.",
    )

    parser.add_argument(
        "--global-search-yaw-rate-rad-s",
        type=float,
        default=0.30,
        help="Yaw rate used when no camera sees the tag.",
    )

    parser.add_argument(
        "--global-search-turn-duration-s",
        type=float,
        default=1.0,
        help="Turn duration used when no camera sees the tag.",
    )

    parser.add_argument(
        "--invert-yaw",
        action="store_true",
        help="Use this if Spot rotates away from the detected target.",
    )

    parser.add_argument(
        "--settle-time-s",
        type=float,
        default=0.35,
        help="Wait time after body movement before next image capture.",
    )

    parser.add_argument(
        "--grasp-offset-px-x",
        type=float,
        default=0.0,
        help="Final grasp pixel offset in image x direction.",
    )

    parser.add_argument(
        "--grasp-offset-px-y",
        type=float,
        default=0.0,
        help="Final grasp pixel offset in image y direction.",
    )

    parser.add_argument(
        "--grasp-offset-tag-x",
        type=float,
        default=0.0,
        help="Offset along AprilTag local x-axis. 1.0 means one tag-width.",
    )

    parser.add_argument(
        "--grasp-offset-tag-y",
        type=float,
        default=0.0,
        help="Offset along AprilTag local y-axis. 1.0 means one tag-height.",
    )

    parser.add_argument(
        "--walk-gaze-mode",
        default="NO_AUTO_WALK_OR_GAZE",
        choices=list(WALK_GAZE_MODE_MAP.keys()),
        help="Manipulation API walk/gaze mode.",
    )

    parser.add_argument(
        "--final-reacquire-attempts",
        type=int,
        default=4,
        help="Attempts to re-detect tag in hand camera immediately before grasp.",
    )

    parser.add_argument(
        "--grasp-timeout-s",
        type=float,
        default=30.0,
        help="Timeout for manipulation feedback.",
    )

    parser.add_argument(
        "--take-lease",
        action="store_true",
        help="Take body lease if already claimed.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search and align only. Do not grasp.",
    )

    parser.add_argument(
        "--hold-seconds-s",
        type=float,
        default=3.0,
        help="How long to hold object after grasp attempt.",
    )

    parser.add_argument(
        "--keep-holding",
        action="store_true",
        help="Do not open gripper after grasp.",
    )

    parser.add_argument(
        "--no-stow",
        action="store_true",
        help="Do not stow arm at end.",
    )

    parser.add_argument(
        "--return-to-start",
        action="store_true",
        help="Walk back to recorded starting pose.",
    )

    parser.add_argument(
        "--return-timeout-s",
        type=float,
        default=15.0,
        help="Timeout for return-to-start command.",
    )

    parser.add_argument(
        "--power-off-at-end",
        action="store_true",
        help="Power off Spot at end.",
    )

    parser.add_argument("--force-top-down-grasp", action="store_true")
    parser.add_argument("--force-horizontal-grasp", action="store_true")
    parser.add_argument("--force-45-angle-grasp", action="store_true")
    parser.add_argument("--force-squeeze-grasp", action="store_true")

    parser.add_argument(
        "--grasp-constraint-tolerance-rad",
        type=float,
        default=0.17,
        help="Orientation constraint tolerance.",
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

    except OperatorAbort:
        logger = bosdyn.client.util.get_logger()
        logger.warning("Script aborted by operator.")
        return False

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
