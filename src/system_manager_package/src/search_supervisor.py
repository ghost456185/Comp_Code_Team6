#!/usr/bin/env python3
"""Search Supervisor — Action Server for Wandering While Searching.

YOUR TASK:
    Implement the search behavior for the WSKR robot. This node is an
    action server that the state manager calls when it needs to find
    something — either a toy (via YOLO detections) or the drop box
    (via ArUco marker detections).

    When a goal arrives, the robot should:
      1. Enable the autopilot (so the robot drives toward headings you set)
      2. Wander by publishing heading angles to WSKR/heading_to_target
      3. Monitor sensor streams for the target
      4. Return success when the target is detected, or abort on timeout

HOW MOVEMENT WORKS:
    You do NOT publish cmd_vel directly. Instead:
      - Publish a heading (degrees) to WSKR/heading_to_target
      - Enable the autopilot via WSKR/autopilot/enable (Bool)
    The autopilot MLP converts the heading + whisker readings into safe
    cmd_vel commands that avoid obstacles.

    Positive heading = target is to the right.
    Negative heading = target is to the left.
    Zero heading     = target is straight ahead.

AVAILABLE SENSOR STREAMS (subscribe to these):
    WSKR/floor_mask       (Image)             — binary mask: white=floor, black=obstacle
    WSKR/aruco_markers    (Float32MultiArray)  — detected ArUco markers [id,x0,y0,...x3,y3]
    WSKR/whisker_lengths  (Float32MultiArray)  — 11 virtual whisker distances (mm)
    vision/yolo/detections (ImgDetectionData)  — YOLO object detections with tracking

CONTROL OUTPUTS (publish to these):
    WSKR/heading_to_target  (Float32)  — desired heading in degrees
    WSKR/autopilot/enable   (Bool)     — enable/disable the autopilot

ACTION INTERFACE:
    Action name: WSKR/search_behavior
    Type: WskrSearch (robot_interfaces/action/WskrSearch)

    Goal fields:
        uint8  target_type   — TARGET_TOY (0) or TARGET_BOX (1)
        uint8  target_id     — ArUco marker ID (only used for TARGET_BOX)
        float32 timeout_sec  — max search duration

    Result fields:
        bool success
        ImgDetectionData detected_object  — the detection that was found

    Feedback fields:
        float32 elapsed_sec
        string  current_phase
        int32   detections_sampled

================================================================================
MINI-TUTORIAL: Writing an action server with sensor subscriptions
================================================================================

1) Cache sensor data in subscription callbacks (thread-safe with a lock):

    def _on_whiskers(self, msg):
        with self.lock:
            self.latest_whiskers = np.asarray(msg.data)

2) In your search loop, read the cached data:

    with self.lock:
        whiskers = self.latest_whiskers

    if whiskers is not None:
        left = float(np.mean(whiskers[:5]))
        right = float(np.mean(whiskers[6:]))
        if left < 200.0:
            heading = -30.0   # obstacle on left, steer right
        elif right < 200.0:
            heading = 30.0    # obstacle on right, steer left

3) Publish the heading to make the robot turn:

    msg = Float32()
    msg.data = heading
    self.heading_pub.publish(msg)

4) Check for YOLO detections (toy search):

    with self.lock:
        detections = self.latest_detections

    if detections is not None:
        for conf in detections.confidence:
            if conf >= self.conf_threshold:
                return detections  # found it!

5) Check for ArUco markers (box search):

    with self.lock:
        markers = self.latest_aruco_markers

    if markers is not None and len(markers.data) >= 9:
        for i in range(0, len(markers.data), 9):
            marker_id = int(markers.data[i])
            if marker_id == target_id:
                return marker_id   # found the box!

6) Use time.sleep() for polling (NOT asyncio.sleep — see threading note):

    while time.time() - start_time < timeout_sec:
        # ... check sensors, update heading ...
        time.sleep(0.05)

================================================================================
"""
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray

from robot_interfaces.action import WskrSearch
from robot_interfaces.msg import ImgDetectionData
from system_manager_package.constants import (
    SEARCH_ARUCO_ID,
    SEARCH_ARUCO_STALE_TIMEOUT_SEC,
    SEARCH_CONFIDENCE_THRESHOLD,
    SEARCH_FLOOR_OBSTACLE_RATIO,
    SEARCH_HEADING_CHANGE_PERIOD_SEC,
    SEARCH_LOOK_DURATION_SEC,
    SEARCH_MAX_HEADING_ANGLE_DEG,
    SEARCH_POLL_INTERVAL_SEC,
    SEARCH_TIMEOUT_SEC,
    SEARCH_WANDER_SPEED_MPS,
)


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class SearchBehavior(Node):
    """Wander + search behavior action server."""

    TARGET_TOY = 0
    TARGET_BOX = 1

    def __init__(self) -> None:
        super().__init__('search_supervisor')

        # Thread-safe sensor cache
        self.lock = threading.Lock()
        self.latest_whiskers: Optional[np.ndarray] = None
        self.latest_detections: Optional[ImgDetectionData] = None
        self.latest_floor_mask: Optional[np.ndarray] = None
        self.latest_aruco_markers: Optional[Float32MultiArray] = None
        self.latest_aruco_markers_time: float = 0.0
        self.latest_floor_mask_time: float = 0.0

        # Search tuning sourced from constants.py so tuning stays centralized.
        self.search_wander_speed_mps = SEARCH_WANDER_SPEED_MPS
        self.search_look_duration_sec = SEARCH_LOOK_DURATION_SEC
        self.search_confidence_threshold = SEARCH_CONFIDENCE_THRESHOLD
        self.search_poll_interval_sec = SEARCH_POLL_INTERVAL_SEC
        self.search_floor_obstacle_ratio = SEARCH_FLOOR_OBSTACLE_RATIO
        self.search_aruco_stale_timeout_sec = SEARCH_ARUCO_STALE_TIMEOUT_SEC
        self.search_heading_change_period_sec = SEARCH_HEADING_CHANGE_PERIOD_SEC
        self.search_max_heading_angle_deg = SEARCH_MAX_HEADING_ANGLE_DEG
        self.search_aruco_id = SEARCH_ARUCO_ID
        self.search_timeout_sec = SEARCH_TIMEOUT_SEC

        cb_group = ReentrantCallbackGroup()

        # ── Sensor subscriptions ────────────────────────────────────
        self.create_subscription(
            Image, 'WSKR/floor_mask', self._on_floor_mask,
            IMAGE_QOS, callback_group=cb_group,
        )
        self.create_subscription(
            Float32MultiArray, 'WSKR/aruco_markers', self._on_aruco_markers,
            10, callback_group=cb_group,
        )
        self.create_subscription(
            Float32MultiArray, 'WSKR/whisker_lengths', self._on_whiskers,
            10, callback_group=cb_group,
        )
        self.create_subscription(
            ImgDetectionData, 'vision/yolo/detections', self._on_detections,
            IMAGE_QOS, callback_group=cb_group,
        )

        # ── Control outputs ─────────────────────────────────────────
        self.heading_pub = self.create_publisher(
            Float32, 'WSKR/heading_to_target', 10,
        )
        autopilot_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.autopilot_enable_pub = self.create_publisher(
            Bool, 'WSKR/autopilot/enable', autopilot_qos,
        )

        # ── Action server ───────────────────────────────────────────
        self._action_server = ActionServer(
            self,
            WskrSearch,
            'WSKR/search_behavior',
            execute_callback=self._execute_search,
            cancel_callback=self._handle_cancel,
            callback_group=cb_group,
        )

        self.get_logger().info('Search supervisor action server ready.')

    # ================================================================
    #  Sensor callbacks — cache latest data (thread-safe)
    # ================================================================

    def _on_floor_mask(self, msg: Image) -> None:
        mask = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width,
        )
        with self.lock:
            self.latest_floor_mask = mask
            self.latest_floor_mask_time = time.time()

    def _on_aruco_markers(self, msg: Float32MultiArray) -> None:
        with self.lock:
            self.latest_aruco_markers = msg
            self.latest_aruco_markers_time = time.time()

    def _on_whiskers(self, msg: Float32MultiArray) -> None:
        with self.lock:
            self.latest_whiskers = np.asarray(msg.data, dtype=np.float64)

    def _on_detections(self, msg: ImgDetectionData) -> None:
        with self.lock:
            self.latest_detections = msg

    # ================================================================
    #  Helpers — use these to control the robot
    # ================================================================

    def _handle_cancel(self, goal_handle) -> CancelResponse:
        self.get_logger().info('Search goal cancellation received.')
        return CancelResponse.ACCEPT

    def _publish_heading(self, heading_deg: float) -> None:
        """Send a heading angle to the autopilot."""
        msg = Float32()
        msg.data = float(heading_deg)
        self.heading_pub.publish(msg)

    def _enable_autopilot(self, enabled: bool) -> None:
        """Turn the autopilot on or off."""
        msg = Bool()
        msg.data = bool(enabled)
        self.autopilot_enable_pub.publish(msg)

    def _stop_robot(self) -> None:
        """Zero heading and disable autopilot."""
        self._publish_heading(0.0)
        self._enable_autopilot(False)

    # ================================================================
    #  Main action callback — implement your search logic here
    # ================================================================

    async def _execute_search(self, goal_handle) -> WskrSearch.Result:
        """Called when the state manager sends a search goal.

        Args:
            goal_handle: contains .request with fields:
                - target_type: TARGET_TOY (0) or TARGET_BOX (1)
                - target_id:   ArUco ID (for box search)
                - timeout_sec: max search time

        Returns:
            WskrSearch.Result with .success and .detected_object
        """
        goal = goal_handle.request
        target_type = goal.target_type
        target_id = int(goal.target_id) if int(goal.target_id) != 0 else int(self.search_aruco_id)
        timeout_sec = float(goal.timeout_sec) if goal.timeout_sec > 0 else float(self.search_timeout_sec)

        self.get_logger().info(
            f'Search started: target={"TOY" if target_type == self.TARGET_TOY else "BOX"}, '
            f'id={target_id}, timeout={timeout_sec}s'
        )

        # Implementation parameters
        conf_threshold = float(self.search_confidence_threshold)
        whisker_obstacle_thresh = 200.0
        poll_sleep = float(self.search_poll_interval_sec)
        look_duration_sec = float(self.search_look_duration_sec)
        heading_change_period_sec = float(self.search_heading_change_period_sec)
        wander_amplitude = float(self.search_max_heading_angle_deg)
        stale_aruco_timeout_sec = float(self.search_aruco_stale_timeout_sec)
        floor_obstacle_ratio = float(self.search_floor_obstacle_ratio)

        start_time = time.time()
        feedback = WskrSearch.Feedback()
        detections_sampled = 0

        # enable autopilot
        self._enable_autopilot(True)

        # helper detection checks
        def _check_for_toy():
            nonlocal detections_sampled
            with self.lock:
                det = self.latest_detections
            if det is None:
                return None
            detections_sampled += 1
            # confidences may be empty — iterate
            for conf in getattr(det, 'confidence', []) or []:
                if conf >= conf_threshold:
                    return det
            return None

        def _check_for_box(tid: int):
            with self.lock:
                markers = self.latest_aruco_markers
                markers_time = self.latest_aruco_markers_time
            if markers is None or not getattr(markers, 'data', None):
                return None
            if time.time() - markers_time > stale_aruco_timeout_sec:
                return None
            data = markers.data
            # markers use 9 entries per marker: id,x0,y0,...x3,y3
            if len(data) < 9:
                return None
            for i in range(0, len(data), 9):
                try:
                    marker_id = int(data[i])
                except Exception:
                    continue
                if marker_id == tid:
                    # create a minimal ImgDetectionData to return
                    det = ImgDetectionData()
                    det.image_width = 0
                    det.image_height = 0
                    det.inference_time = 0.0
                    det.detection_ids = [str(marker_id)]
                    det.x = [float(data[i+1])]
                    det.y = [float(data[i+2])]
                    det.width = [0.0]
                    det.height = [0.0]
                    det.distance = [0.0]
                    det.location = []
                    det.yaw = [0.0]
                    det.class_name = ["aruco_box"]
                    det.confidence = [1.0]
                    det.aspect_ratio = [0.0]
                    return det
            return None

        def _floor_obstacle_detected() -> bool:
            with self.lock:
                floor_mask = None if self.latest_floor_mask is None else np.copy(self.latest_floor_mask)
                floor_time = self.latest_floor_mask_time
            if floor_mask is None or floor_mask.size == 0:
                return False
            if floor_time > 0.0 and (time.time() - floor_time) > stale_aruco_timeout_sec:
                return False
            height, width = floor_mask.shape[:2]
            sample_w = max(1, int(width * 0.5))
            sample_h = max(1, int(height * 0.25))
            x0 = max(0, (width - sample_w) // 2)
            y0 = max(0, height - sample_h)
            sample = floor_mask[y0:y0 + sample_h, x0:x0 + sample_w]
            if sample.size == 0:
                return False
            non_floor_ratio = float(np.count_nonzero(sample < 128)) / float(sample.size)
            return non_floor_ratio >= floor_obstacle_ratio

        # wander variables (triangle wave)
        wander_period = max(heading_change_period_sec, 0.1)

        try:
            while time.time() - start_time < timeout_sec:
                # check cancel
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result = WskrSearch.Result()
                    result.success = False
                    return result

                elapsed = time.time() - start_time

                # compute heading using whiskers if available
                heading = 0.0
                phase = 'looking'
                with self.lock:
                    whiskers = None if self.latest_whiskers is None else np.copy(self.latest_whiskers)
                cycle_time = elapsed % max(look_duration_sec + wander_period, 0.1)
                if cycle_time < look_duration_sec:
                    phase = 'looking'
                    heading = 0.0
                elif whiskers is not None and whiskers.size >= 11:
                    left = float(np.mean(whiskers[:5]))
                    right = float(np.mean(whiskers[6:11]))
                    if left < whisker_obstacle_thresh:
                        heading = -wander_amplitude
                        phase = 'avoiding_left'
                    elif right < whisker_obstacle_thresh:
                        heading = wander_amplitude
                        phase = 'avoiding_right'
                    else:
                        # forward wander small oscillation
                        phase = 'wandering'
                        # triangle wave between -amplitude..+amplitude
                        t = ((cycle_time - look_duration_sec) % wander_period) / wander_period
                        if t < 0.5:
                            heading = -wander_amplitude + (t / 0.5) * (2 * wander_amplitude)
                        else:
                            heading = wander_amplitude - ((t - 0.5) / 0.5) * (2 * wander_amplitude)
                else:
                    # no whiskers — deterministic wander
                    phase = 'wandering'
                    t = ((cycle_time - look_duration_sec) % wander_period) / wander_period
                    if t < 0.5:
                        heading = -wander_amplitude + (t / 0.5) * (2 * wander_amplitude)
                    else:
                        heading = wander_amplitude - ((t - 0.5) / 0.5) * (2 * wander_amplitude)

                if _floor_obstacle_detected():
                    heading = -wander_amplitude if int(elapsed) % 2 == 0 else wander_amplitude
                    phase = 'floor_obstacle'

                # publish heading
                self._publish_heading(heading)

                # publish feedback occasionally
                feedback.elapsed_sec = float(elapsed)
                feedback.current_phase = phase
                feedback.detections_sampled = int(detections_sampled)
                try:
                    goal_handle.publish_feedback(feedback)
                except Exception:
                    pass

                # check target
                if target_type == self.TARGET_TOY:
                    found = _check_for_toy()
                    if found is not None:
                        result = WskrSearch.Result()
                        result.success = True
                        result.detected_object = found
                        goal_handle.succeed()
                        return result
                else:
                    found = _check_for_box(target_id)
                    if found is not None:
                        result = WskrSearch.Result()
                        result.success = True
                        result.detected_object = found
                        goal_handle.succeed()
                        return result

                time.sleep(poll_sleep)

            # timeout
            self.get_logger().info('Search timed out')
            goal_handle.abort()
            result = WskrSearch.Result()
            result.success = False
            return result
        finally:
            # ensure robot stops
            self._stop_robot()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SearchBehavior()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
