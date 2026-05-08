#!/usr/bin/env python3
"""Object selection: (class priority, -y) -> best single detection.

Subscribes to ``vision/yolo/detections`` and caches the latest frame.
Exposes the selection two ways:

  1. ``select_object_service`` (SelectObject, trigger-style) — returns the
     best pick from the most recent cached YOLO frame.  The state_manager
     calls this during the SELECT state.
  2. ``vision/selected_object`` (continuous stream) — republishes the
     current best pick on every incoming frame for dashboard overlays.

Grace-period behavior for close-range vision loss:
  - If YOLO detections drop out (empty frame), the node tolerates up to
    VISION_DROPOUT_GRACE_PERIOD_S of silence. During this window, it keeps
    republishing the last stable selection so downstream doesn't abort.
  - After the grace period expires, it publishes empty and resets state.
"""
from typing import Optional
import time

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from robot_interfaces.msg import ImgDetectionData
from robot_interfaces.srv import SelectObject

from system_manager_package.constants import (
    SELECTION_CLASS_PRIORITIES,
    SELECTION_MIN_CONFIDENCE,
    VISION_DEBOUNCE_FRAMES,
    VISION_DROPOUT_GRACE_PERIOD_S,
)


STREAM_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class ObjectSelection(Node):
    def __init__(self):
        super().__init__('object_selection_node')

        self.declare_parameter(
            'class_priorities',
            list(SELECTION_CLASS_PRIORITIES),
        )
        self.declare_parameter('min_confidence', SELECTION_MIN_CONFIDENCE)
        self.declare_parameter('debounce_frames', VISION_DEBOUNCE_FRAMES)

        self.class_priorities = [
            str(c) for c in self.get_parameter('class_priorities').value
        ]
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.debounce_frames = int(self.get_parameter('debounce_frames').value)

        self._latest_detections: Optional[ImgDetectionData] = None
        # Debounce state: track candidate class and counts to avoid flicker
        self._candidate_class: Optional[str] = None
        self._stable_count: int = 0
        self._last_stable_selected: Optional[ImgDetectionData] = None
        
        # Grace-period tracking for close-range vision dropout tolerance.
        # When YOLO detections are empty, we start a timer. If the grace
        # period hasn't expired, we keep republishing the last stable object
        # instead of clearing it immediately.
        self._dropout_grace_period_s = VISION_DROPOUT_GRACE_PERIOD_S
        self._last_empty_frame_time: Optional[float] = None

        self.srv = self.create_service(
            SelectObject, 'select_object_service', self._handle_select_service,
        )

        self.create_subscription(
            ImgDetectionData, 'vision/yolo/detections',
            self._on_detections, STREAM_QOS,
        )
        self.selected_pub = self.create_publisher(
            ImgDetectionData, 'vision/selected_object', STREAM_QOS,
        )

        self.get_logger().info(
            f"ObjectSelection ready (priorities={self.class_priorities}, "
            f"min_confidence={self.min_confidence:.2f}, "
            f"dropout_grace_period={self._dropout_grace_period_s:.1f}s). "
            "Service: select_object_service; stream: vision/selected_object."
        )

    # ------------------------------------------------------------ helpers

    def _priority_rank(self, class_name: str) -> int:
        try:
            return self.class_priorities.index(class_name)
        except ValueError:
            return len(self.class_priorities)

    def _pick_best(self, detections: ImgDetectionData) -> Optional[int]:
        """Return the index of the best detection or None.
        Only detections whose class is in class_priorities are eligible."""
        n = len(detections.x)
        if n == 0:
            return None
        best_idx = -1
        best_key = None
        for i in range(n):
            conf = (
                float(detections.confidence[i])
                if i < len(detections.confidence)
                else 0.0
            )
            if conf < self.min_confidence:
                continue
            cls = (
                str(detections.class_name[i])
                if i < len(detections.class_name)
                else ''
            )
            if cls not in self.class_priorities:
                continue
            key = (self._priority_rank(cls), -float(detections.y[i]))
            if best_key is None or key < best_key:
                best_key = key
                best_idx = i
        return best_idx if best_idx >= 0 else None

    def _extract_single(
        self, detections: ImgDetectionData, idx: int,
    ) -> ImgDetectionData:
        out = ImgDetectionData()
        out.image_width = detections.image_width
        out.image_height = detections.image_height
        out.inference_time = detections.inference_time
        out.x = [detections.x[idx]]
        out.y = [detections.y[idx]]
        out.width = [detections.width[idx]]
        out.height = [detections.height[idx]]
        out.class_name = [detections.class_name[idx]]
        out.confidence = [detections.confidence[idx]]
        out.aspect_ratio = (
            [detections.aspect_ratio[idx]]
            if idx < len(detections.aspect_ratio)
            else [0.0]
        )
        out.detection_ids = (
            [detections.detection_ids[idx]]
            if idx < len(detections.detection_ids)
            else ['']
        )
        out.distance = (
            [detections.distance[idx]]
            if idx < len(detections.distance)
            else [0.0]
        )
        out.location = (
            [detections.location[idx]]
            if idx < len(detections.location)
            else [Point()]
        )
        out.yaw = (
            [detections.yaw[idx]]
            if idx < len(detections.yaw)
            else [0.0]
        )
        return out

    # --------------------------------------------------------- subscription

    def _on_detections(self, msg: ImgDetectionData) -> None:
        """Cache the latest frame and publish the streaming selection.
        
        If detections are empty (idx=None), start/track the dropout grace period.
        While within the grace period, keep republishing the last stable object
        to tolerate brief vision dropouts (e.g., close-range occlusions). After
        the grace period expires, clear selection and reset state.
        """
        self._latest_detections = msg

        idx = self._pick_best(msg)
        if idx is None:
            # No candidate this frame — handle via grace period.
            now = time.time()
            if self._last_empty_frame_time is None:
                # First empty frame in a sequence
                self._last_empty_frame_time = now
                grace_elapsed = 0.0
            else:
                grace_elapsed = now - self._last_empty_frame_time
            
            if grace_elapsed < self._dropout_grace_period_s:
                # Still within grace period — keep republishing the last stable
                # selection if it exists, so downstreams don't see a flicker/abort.
                if self._last_stable_selected is not None:
                    self.selected_pub.publish(self._last_stable_selected)
                    if grace_elapsed == 0.0:
                        self.get_logger().debug(
                            f"Vision dropout detected. Keeping last stable object active "
                            f"({self._dropout_grace_period_s}s grace period)."
                        )
                else:
                    # No prior stable selection to reuse
                    empty = ImgDetectionData()
                    empty.image_width = msg.image_width
                    empty.image_height = msg.image_height
                    self.selected_pub.publish(empty)
            else:
                # Grace period expired — clear everything and publish empty.
                if grace_elapsed < self._dropout_grace_period_s + 1.0:
                    # Only log once per expiry (within 1 sec of the boundary)
                    self.get_logger().warn(
                        f"Vision dropout exceeded grace period ({grace_elapsed:.1f}s > {self._dropout_grace_period_s}s). "
                        "Clearing selection."
                    )
                empty = ImgDetectionData()
                empty.image_width = msg.image_width
                empty.image_height = msg.image_height
                self._candidate_class = None
                self._stable_count = 0
                self._last_stable_selected = None
                self._last_empty_frame_time = None
                self.selected_pub.publish(empty)
            return
        
        # Candidate detected — reset dropout timer and perform debounce by class.
        self._last_empty_frame_time = None
        cls = (
            str(msg.class_name[idx])
            if idx < len(msg.class_name)
            else ''
        )

        if self._candidate_class == cls:
            self._stable_count += 1
        else:
            self._candidate_class = cls
            self._stable_count = 1

        if self._stable_count >= max(1, self.debounce_frames):
            # Candidate is stable — publish the fresh selection and cache it
            selected = self._extract_single(msg, idx)
            self._last_stable_selected = selected
            self.selected_pub.publish(selected)
        else:
            # Not stable yet — publish the last stable selection if present,
            # otherwise publish an empty placeholder so downstreams don't flip.
            if self._last_stable_selected is not None:
                self.selected_pub.publish(self._last_stable_selected)
            else:
                empty = ImgDetectionData()
                empty.image_width = msg.image_width
                empty.image_height = msg.image_height
                self.selected_pub.publish(empty)

    # ------------------------------------------------------------- service

    def _handle_select_service(self, _request, response):
        """Pick the best object from the latest cached YOLO frame."""
        if self._latest_detections is None:
            self.get_logger().warn("SelectObject called but no detections received yet.")
            response.success = False
            return response

        detections = self._latest_detections
        self.get_logger().info(f"SelectObject service: n={len(detections.x)}")

        idx = self._pick_best(detections)
        if idx is None:
            self.get_logger().warn(
                f"No detection passed min_confidence={self.min_confidence:.2f}."
            )
            response.success = False
            return response

        selected = self._extract_single(detections, idx)
        self.get_logger().info(
            f"Selected idx={idx} class={selected.class_name[0]} "
            f"y={selected.y[0]:.1f} conf={selected.confidence[0]:.2f} "
            f"track_id={selected.detection_ids[0]}"
        )
        response.success = True
        response.selected_obj = selected
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ObjectSelection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
