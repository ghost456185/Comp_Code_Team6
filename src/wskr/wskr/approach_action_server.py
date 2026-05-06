"""WSKR Approach Action Server — "drive up to this specific thing."

This is the supervisor of the approach behavior. A client (typically a GUI
like ``Start_Aruco_Approach``) sends an ``ApproachObject`` action goal
naming an ArUco tag ID (or a toy bbox seeded by an external detector).
This node then:

    1. Runs vision every frame — ArUco detection for TARGET_BOX goals or a
       CSRT template tracker for TARGET_TOY. Publishes a *visual observation*
       of the target's heading whenever detection succeeds.
    2. Enables the autopilot (``wskr_autopilot``) for the duration of the
       goal. The autopilot owns ``WSKR/cmd_vel`` and consumes the fused
       heading + whiskers itself.
    3. Watches whisker drive-distances + the fused heading to decide when
       the target has been reached, aborts on timeout / loss / reacquisition
       failure, and disables the autopilot when the goal ends.

Topics:
    subscribes  camera1/image_raw/compressed     — main camera (JPEG).
    subscribes  WSKR/whisker_lengths             — 11 floats (drive distances).
    subscribes  WSKR/heading_to_target           — fused heading from DR node.
    subscribes  WSKR/tracking_mode               — ``visual`` or ``dead_reckoning``.
    publishes   WSKR/heading_to_target/visual_obs — heading from bbox (when seen).
    publishes   WSKR/tracked_bbox                — width-normalized (x,y,w,h).
    publishes   WSKR/autopilot/enable            — latched Bool gating the autopilot.
    publishes   WSKR/cmd_vel                     — zero Twist on goal end (safety stop).
Action:
    server of   WSKR/approach_object             — ApproachObject.action.
"""
import math
import threading
import time
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Empty, Float32, Float32MultiArray, MultiArrayDimension, String

from robot_interfaces.action import ApproachObject
from robot_interfaces.msg import ApproachTargetInfo, ImgDetectionData, TrackedBbox

from .lens_model import LensParams, compute_heading_rad

from system_manager_package.constants import (
    APPROACH_ARUCO_DETECT_SCALE,
    APPROACH_CLASS_CHANGE_ABORT_SEC,
    APPROACH_PROXIMITY_SUCCESS_MM,
    APPROACH_REACQUIRE_FAILURE_DEG,
    APPROACH_REACQUIRE_FAILURE_FRAMES,
    APPROACH_REACQUIRE_THRESHOLD,
    APPROACH_SLOW_FRAME_WARN_MS,
    APPROACH_TARGET_LOST_TIMEOUT_SEC,
    APPROACH_TIMEOUT_SEC,
    APPROACH_TRACK_IOU_HANDOFF,
    APPROACH_YOLO_GAP_ABORT_SEC,
    APPROACH_YOLO_STALENESS_WARN_SEC,
)


IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

LENS_PARAM_NAMES = ('x_min', 'x_max', 'cy', 'hfov_deg', 'tilt_deg', 'y_offset')


class WSKRApproachActionServer(Node):
    def __init__(self) -> None:
        super().__init__('wskr_approach_action')

        self.bridge = CvBridge()

        self.declare_parameter('aruco_id', 1)
        self.declare_parameter('approach_timeout_sec', APPROACH_TIMEOUT_SEC)
        self.declare_parameter('proximity_success_mm', APPROACH_PROXIMITY_SUCCESS_MM)
        self.declare_parameter('target_lost_timeout_sec', APPROACH_TARGET_LOST_TIMEOUT_SEC)
        self.declare_parameter('reacquire_threshold', APPROACH_REACQUIRE_THRESHOLD)
        self.declare_parameter('reacquire_failure_deg', APPROACH_REACQUIRE_FAILURE_DEG)
        # Number of consecutive frames without a valid detection before the
        # reacquire-failure abort fires. Needs to be large enough to ride out
        # transient motion blur / occlusion; too small and moving the marker
        # aborts the goal.
        self.declare_parameter('reacquire_failure_frames', APPROACH_REACQUIRE_FAILURE_FRAMES)
        self.declare_parameter('aruco_detect_scale', APPROACH_ARUCO_DETECT_SCALE)
        self.declare_parameter('slow_frame_warn_ms', APPROACH_SLOW_FRAME_WARN_MS)

        # YOLO/CSRT fusion parameters (TOY target only).
        #
        # yolo_gap_abort_sec — hard cap on how long we coast on CSRT without a
        #     same-class YOLO match. When exceeded, the goal aborts so the
        #     state machine falls back to WANDER.
        # class_change_abort_sec — same idea but for "only different-class
        #     detections visible." Protects against CSRT drifting onto
        #     something with a different label.
        # track_iou_handoff_threshold — minimum IoU(csrt_bbox, yolo_bbox) for
        #     accepting a new track id on the same class. ByteTrack issues
        #     new ids after brief occlusions; this lets us adopt them when
        #     CSRT confirms they're the same object.
        # yolo_staleness_warn_sec — log-only; does not gate fusion.
        self.declare_parameter('yolo_gap_abort_sec', APPROACH_YOLO_GAP_ABORT_SEC)
        self.declare_parameter('class_change_abort_sec', APPROACH_CLASS_CHANGE_ABORT_SEC)
        self.declare_parameter('track_iou_handoff_threshold', APPROACH_TRACK_IOU_HANDOFF)
        self.declare_parameter('yolo_staleness_warn_sec', APPROACH_YOLO_STALENESS_WARN_SEC)
        
        # Alignment parameters: stop and center before declaring proximity success
        self.declare_parameter('approach_align_deadband_deg', 5.0)
        self.declare_parameter('approach_align_kp', 0.75) # P gain for alignment turn rate
        self.declare_parameter('approach_align_rate_hz', 10.0)
        self.declare_parameter('approach_align_timeout_s', 3.0)
        self.declare_parameter('approach_align_max_rate_rad_s', 0.5)

        # Lens params (normalized, for in-process heading computation).
        _lp = LensParams()
        self.declare_parameter('x_min', _lp.x_min)
        self.declare_parameter('x_max', _lp.x_max)
        self.declare_parameter('cy', _lp.cy)
        self.declare_parameter('hfov_deg', _lp.hfov_deg)
        self.declare_parameter('tilt_deg', _lp.tilt_deg)
        self.declare_parameter('y_offset', _lp.y_offset)

        self.aruco_id = int(self.get_parameter('aruco_id').value)
        self.approach_timeout_sec = float(self.get_parameter('approach_timeout_sec').value)
        self.proximity_success_mm = float(self.get_parameter('proximity_success_mm').value)
        self.target_lost_timeout_sec = float(self.get_parameter('target_lost_timeout_sec').value)
        self.reacquire_threshold = float(self.get_parameter('reacquire_threshold').value)
        self.reacquire_failure_deg = float(self.get_parameter('reacquire_failure_deg').value)
        self.reacquire_failure_frames = int(self.get_parameter('reacquire_failure_frames').value)
        self.aruco_detect_scale = float(self.get_parameter('aruco_detect_scale').value)
        self.slow_frame_warn_ms = float(self.get_parameter('slow_frame_warn_ms').value)
        self.yolo_gap_abort_sec = float(self.get_parameter('yolo_gap_abort_sec').value)
        self.class_change_abort_sec = float(self.get_parameter('class_change_abort_sec').value)
        self.track_iou_handoff_threshold = float(self.get_parameter('track_iou_handoff_threshold').value)
        self.yolo_staleness_warn_sec = float(self.get_parameter('yolo_staleness_warn_sec').value)
        
        # Alignment params
        self._align_deadband_deg = float(self.get_parameter('approach_align_deadband_deg').value)
        self._align_kp = float(self.get_parameter('approach_align_kp').value)
        self._align_rate_hz = float(self.get_parameter('approach_align_rate_hz').value)
        self._align_timeout_s = float(self.get_parameter('approach_align_timeout_s').value)
        self._align_max_rate_rad_s = float(self.get_parameter('approach_align_max_rate_rad_s').value)

        self._lens_lock = threading.Lock()
        self._lens_params = self._read_lens_params()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        cb_group = ReentrantCallbackGroup()

        self.image_sub = self.create_subscription(
            CompressedImage, 'camera1/image_raw/compressed', self.image_callback, IMAGE_QOS,
            callback_group=cb_group,
        )
        self.whisker_sub = self.create_subscription(
            Float32MultiArray, 'WSKR/whisker_lengths', self.whisker_callback, 10, callback_group=cb_group
        )
        self.target_whisker_sub = self.create_subscription(
            Float32MultiArray, 'WSKR/target_whisker_lengths', self.target_whisker_callback, 10, callback_group=cb_group
        )

        # Visual observations only — the dead_reckoning_node owns the fused topic.
        self.visual_obs_pub = self.create_publisher(
            Float32, 'WSKR/heading_to_target/visual_obs', 10
        )
        # Latched enable to gate the autopilot for the duration of a goal.
        autopilot_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.autopilot_enable_pub = self.create_publisher(
            Bool, 'WSKR/autopilot/enable', autopilot_qos
        )
        # Safety-stop publisher: zero Twist when a goal ends, in case the
        # autopilot is slow to react to the disable flag.
        self.cmd_pub = self.create_publisher(Twist, 'WSKR/cmd_vel', 10)
        self.stop_pub = self.create_publisher(Empty, 'WSKR/stop', 1)
        self.tracked_bbox_pub = self.create_publisher(TrackedBbox, 'WSKR/tracked_bbox', 10)
        # Latched target-info publisher — dashboards / consumers use this to
        # filter the detection stream to just the object being approached.
        # TRANSIENT_LOCAL so late subscribers see the current target state.
        target_info_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.target_info_pub = self.create_publisher(
            ApproachTargetInfo, 'WSKR/approach_target_info', target_info_qos,
        )
        # All detected ArUco markers, width-normalized, published whenever a
        # subscriber is present (e.g. the Foxglove dashboard). Layout is
        # flat: data = [id, x1_n, y1_n, x2_n, y2_n, x3_n, y3_n, x4_n, y4_n, ...]
        # with dim[0].label=='marker' size N stride 9, dim[1].label=='field'.
        # Normalized by the decoded frame width, same convention as tracked_bbox.
        self.markers_pub = self.create_publisher(Float32MultiArray, 'WSKR/aruco_markers', 10)
        self.yolo_enable_pub = self.create_publisher(Bool, 'WSKR/yolo_streaming_enable', 10)
        self._publish_autopilot_enable(False)
        self._publish_target_info_inactive()

        self.fused_heading_sub = self.create_subscription(
            Float32, 'WSKR/heading_to_target', self._on_fused_heading, 10, callback_group=cb_group
        )
        self.tracking_mode_sub = self.create_subscription(
            String, 'WSKR/tracking_mode', self._on_tracking_mode, 10, callback_group=cb_group
        )
        self.yolo_detections_sub = self.create_subscription(
            ImgDetectionData, 'vision/yolo/detections', self._on_yolo_detections,
            IMAGE_QOS, callback_group=cb_group,
        )

        self.action_server = ActionServer(
            self,
            ApproachObject,
            'WSKR/approach_object',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=cb_group,
        )

        self.lock = threading.Lock()
        self.active_goal_handle = None
        self.goal_target_type = None
        self.goal_object_id = -1
        self.goal_selected_obj = None

        self.latest_whiskers: Optional[np.ndarray] = None
        self.latest_target_whiskers: Optional[np.ndarray] = None
        self.last_heading_deg = 0.0  # mirror of WSKR/heading_to_target (fused)
        self.tracking_mode = 'visual'  # mirror of WSKR/tracking_mode
        self.visually_tracked = True  # bbox present on most recent frame
        self.last_tracked_bbox: Optional[Tuple[int, int, int, int]] = None
        self.frames_since_valid_track = 0
        self.target_lost_threshold_frames = 3

        self.pending_toy_bbox: Optional[Tuple[float, float, float, float]] = None
        self.tracker = None
        self.last_frame = None
        self.lost_since: Optional[float] = None
        self.lost_template: Optional[np.ndarray] = None
        # Per-frame provenance stamp updated by _fuse_yolo_with_csrt (TOY)
        # or _pick_target_bbox_from_detection (BOX). Cleared before each
        # frame's track attempt in image_callback.
        self._tracked_bbox_source: str = ''

        # YOLO stream + fusion state (TOY target). Stream cache is populated
        # anytime regardless of goal; the goal-scoped fields are reset in
        # execute_callback.
        self.latest_yolo_msg: Optional[ImgDetectionData] = None
        self.latest_yolo_t: Optional[float] = None
        self.goal_track_id: int = -1
        self.goal_class_name: str = ''
        self.yolo_match_last_t: Optional[float] = None
        self.class_mismatch_since: Optional[float] = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict)

        self.get_logger().info('Approach action server ready on WSKR/approach_object.')

    def goal_callback(self, goal_request: ApproachObject.Goal) -> GoalResponse:
        """Reject new goals while one is active; reject unknown target types."""
        with self.lock:
            if self.active_goal_handle is not None:
                return GoalResponse.REJECT
        if goal_request.target_type not in (ApproachObject.Goal.TARGET_TOY, ApproachObject.Goal.TARGET_BOX):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _create_csrt(self):
        # Stock defaults with segmentation disabled (segfault trigger on OpenCV 4.10)
        # and a slightly stricter PSR so track loss is reported sooner, handing off
        # to template-match re-acquisition.
        params = cv2.TrackerCSRT_Params()
        params.use_segmentation = False
        params.psr_threshold = 0.06
        return cv2.TrackerCSRT_create(params)

    def _sanitize_bbox(self, bbox, frame_shape):
        x, y, w, h = bbox
        H, W = frame_shape[:2]

        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(1, min(int(w), W - x))
        h = max(1, min(int(h), H - y))

        return (x, y, w, h)

    def _rescale_bbox(self, bbox, src_w: int, src_h: int, dst_w: int, dst_h: int):
        """Rescale bbox from source image dimensions to destination dimensions."""
        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            return bbox
        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        x, y, w, h = bbox
        return (x * sx, y * sy, w * sx, h * sy)

    def _is_valid_seed_bbox(self, bbox, frame_shape) -> bool:
        """Check bbox is strictly inside frame and large enough for CSRT init."""
        x, y, w, h = bbox
        fh, fw = frame_shape[:2]
        if fw <= 0 or fh <= 0:
            return False
        if w < 2 or h < 2:
            return False
        if x < 0 or y < 0:
            return False
        if (x + w) > fw or (y + h) > fh:
            return False
        return True

    def _pad_bbox(self, bbox, frame_shape, pad_frac: float = 0.15):
        x, y, w, h = bbox
        fh, fw = frame_shape[:2]
        pad_w = w * pad_frac
        pad_h = h * pad_frac
        nx = max(0.0, x - pad_w)
        ny = max(0.0, y - pad_h)
        nw = min(fw - nx, w + 2.0 * pad_w)
        nh = min(fh - ny, h + 2.0 * pad_h)
        return self._sanitize_bbox((nx, ny, nw, nh), frame_shape)

    def _extract_bbox_from_selected_obj(self, selected_obj, object_id: int, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
        n = len(selected_obj.x)
        if n == 0:
            return None

        idx = 0
        if n > 1 and selected_obj.detection_ids:
            for i, det_id in enumerate(selected_obj.detection_ids):
                try:
                    if int(det_id) == object_id:
                        idx = i
                        break
                except ValueError:
                    pass

        cx = float(selected_obj.x[idx])
        cy = float(selected_obj.y[idx])
        bw = float(selected_obj.width[idx])
        bh = float(selected_obj.height[idx])

        if cx <= 1.0 and cy <= 1.0 and bw <= 1.0 and bh <= 1.0:
            cx *= w
            cy *= h
            bw *= w
            bh *= h

        x0 = max(0.0, cx - bw / 2.0)
        y0 = max(0.0, cy - bh / 2.0)
        bw = max(2.0, min(bw, w - x0))
        bh = max(2.0, min(bh, h - y0))
        return self._sanitize_bbox((x0, y0, bw, bh), (h, w))

    def whisker_callback(self, msg: Float32MultiArray) -> None:
        """Cache the latest 11-whisker floor-distance array for the control loop."""
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] != 11:
            return
        self.latest_whiskers = data

    def target_whisker_callback(self, msg: Float32MultiArray) -> None:
        """Cache the latest 11-whisker target-bbox distance array for proximity check."""
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] != 11:
            return
        self.latest_target_whiskers = data

    def _read_lens_params(self) -> LensParams:
        gp = self.get_parameter
        return LensParams(
            x_min=float(gp('x_min').value),
            x_max=float(gp('x_max').value),
            cy=float(gp('cy').value),
            hfov_deg=float(gp('hfov_deg').value),
            tilt_deg=float(gp('tilt_deg').value),
            y_offset=float(gp('y_offset').value),
        )

    def _on_set_parameters(self, params) -> SetParametersResult:
        # Live-reconfigure lens params. Any other parameter passes through.
        lens_updates = [p for p in params if p.name in LENS_PARAM_NAMES]
        if lens_updates:
            proposed = {p.name: p.value for p in lens_updates}
            with self._lens_lock:
                current = self._lens_params
                merged = LensParams(
                    x_min=float(proposed.get('x_min', current.x_min)),
                    x_max=float(proposed.get('x_max', current.x_max)),
                    cy=float(proposed.get('cy', current.cy)),
                    hfov_deg=float(proposed.get('hfov_deg', current.hfov_deg)),
                    tilt_deg=float(proposed.get('tilt_deg', current.tilt_deg)),
                    y_offset=float(proposed.get('y_offset', current.y_offset)),
                )
                if merged.x_max <= merged.x_min:
                    return SetParametersResult(successful=False, reason='x_max must be > x_min')
                self._lens_params = merged

        return SetParametersResult(successful=True)

    def _publish_autopilot_enable(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self.autopilot_enable_pub.publish(msg)

    def _publish_target_info(
        self, class_name: str, track_id: int, target_type: int, active: bool,
    ) -> None:
        msg = ApproachTargetInfo()
        msg.class_name = str(class_name)
        msg.track_id = int(track_id)
        msg.target_type = int(target_type)
        msg.active = bool(active)
        self.target_info_pub.publish(msg)

    def _publish_target_info_inactive(self) -> None:
        self._publish_target_info('', -1, 0, False)

    def _compute_and_publish_heading(self, u_norm: float, v_norm: float) -> None:
        """Synchronous heading computation via the in-process lens model.
        Inputs are in width-normalized image coords (u_px / W, v_px / W).
        """
        with self._lens_lock:
            params = self._lens_params
        heading_deg = math.degrees(compute_heading_rad(float(u_norm), float(v_norm), params))
        self._publish_visual_obs(heading_deg)

    def _check_bbox_impinges_whiskers(self, heading_deg: float, whiskers: Optional[np.ndarray], threshold_mm: float = 250.0) -> bool:
        """
        Check if object at heading_deg impinges on whiskers within threshold_mm.
        Maps heading angle to closest whisker and checks its length.
        """
        if whiskers is None or len(whiskers) == 0:
            return False
        
        # Whiskers arranged in fan from -90 to +90 degrees (11 whiskers typical)
        num_whiskers = len(whiskers)
        whisker_angles = np.linspace(-90.0, 90.0, num_whiskers)
        
        # Find closest whisker to object heading
        idx = np.argmin(np.abs(whisker_angles - heading_deg))
        closest_whisker_mm = float(whiskers[idx])
        
        return closest_whisker_mm <= threshold_mm

    def _publish_visual_obs(self, heading_deg: float) -> None:
        msg = Float32()
        msg.data = float(heading_deg)
        self.visual_obs_pub.publish(msg)

    def _on_fused_heading(self, msg: Float32) -> None:
        self.last_heading_deg = float(msg.data)

    def _on_tracking_mode(self, msg: String) -> None:
        self.tracking_mode = msg.data

    def _cache_template(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> None:
        x, y, w, h = self._sanitize_bbox(bbox, frame.shape)
        crop = frame[y:y + h, x:x + w]
        if crop.size > 0:
            self.lost_template = crop.copy()

    def _template_reacquire(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        if self.lost_template is None:
            return None
        th, tw = self.lost_template.shape[:2]
        fh, fw = frame.shape[:2]
        if tw >= fw or th >= fh:
            return None
        result = cv2.matchTemplate(frame, self.lost_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < self.reacquire_threshold:
            return None
        return (float(max_loc[0]), float(max_loc[1]), float(tw), float(th))

    def _on_yolo_detections(self, msg: ImgDetectionData) -> None:
        # Single-reference assignment — no lock needed. Callers read their own
        # local snapshot.
        self.latest_yolo_msg = msg
        self.latest_yolo_t = time.time()

    @staticmethod
    def _iou(bbox_a: Tuple[float, float, float, float],
             bbox_b: Tuple[float, float, float, float]) -> float:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        union = float(aw) * float(ah) + float(bw) * float(bh) - inter
        return inter / union if union > 0.0 else 0.0

    def _reseed_csrt(self, frame: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> None:
        padded = self._pad_bbox(bbox, frame.shape, pad_frac=0.15)
        seed = self._sanitize_bbox(padded, frame.shape)
        if not self._is_valid_seed_bbox(seed, frame.shape):
            return
        try:
            new_tracker = self._create_csrt()
            if new_tracker.init(frame, seed):
                self.tracker = new_tracker
                self.lost_since = None
                self._cache_template(frame, seed)
        except Exception as exc:
            self.get_logger().warn(f'CSRT re-seed failed: {exc}')

    def _fuse_yolo_with_csrt(
        self,
        frame: np.ndarray,
        csrt_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Reconcile the freshest YOLO frame with the CSRT track.

        Rules (TARGET_TOY only):
          - same class + same track_id → authoritative, re-seed CSRT.
          - same class + different track_id + IoU(csrt, yolo) ≥ threshold
                → adopt new track_id, re-seed CSRT.
          - detections present but none of our class → start class-change
                watchdog (abort fires in _check_fusion_timeouts).
          - no usable YOLO match → fall through to CSRT bbox (may be None).
        """
        now = time.time()
        msg = self.latest_yolo_msg
        msg_t = self.latest_yolo_t
        if msg is None or msg_t is None:
            return csrt_bbox

        fh, fw = frame.shape[:2]
        src_w = int(msg.image_width) if msg.image_width else fw
        src_h = int(msg.image_height) if msg.image_height else fh
        sx = float(fw) / float(src_w) if src_w > 0 else 1.0
        sy = float(fh) / float(src_h) if src_h > 0 else 1.0

        same_id_bbox: Optional[Tuple[int, int, int, int]] = None
        same_class_others: list = []
        saw_any_class = False
        saw_target_class = False

        n = len(msg.x)
        for i in range(n):
            cls = msg.class_name[i] if i < len(msg.class_name) else ''
            if not cls:
                continue
            saw_any_class = True
            if cls != self.goal_class_name:
                continue
            saw_target_class = True

            det_id_str = msg.detection_ids[i] if i < len(msg.detection_ids) else ''
            try:
                det_id = int(det_id_str)
            except (TypeError, ValueError):
                # 'tmp-N' or empty — unconfirmed track, ignored for matching.
                continue

            cx = float(msg.x[i])
            cy = float(msg.y[i])
            bw = float(msg.width[i])
            bh = float(msg.height[i])
            x0 = (cx - bw / 2.0) * sx
            y0 = (cy - bh / 2.0) * sy
            yolo_bbox = self._sanitize_bbox(
                (x0, y0, bw * sx, bh * sy), frame.shape
            )

            if det_id == self.goal_track_id:
                same_id_bbox = yolo_bbox
            else:
                same_class_others.append((det_id, yolo_bbox))

        # Same class + same track_id: authoritative.
        if same_id_bbox is not None:
            self._reseed_csrt(frame, same_id_bbox)
            self.yolo_match_last_t = now
            self.class_mismatch_since = None
            self.visually_tracked = True
            self._tracked_bbox_source = 'yolo'
            return same_id_bbox

        # Same class, different track_id: IoU handoff against CSRT.
        if same_class_others and csrt_bbox is not None:
            best_iou = 0.0
            best_det_id = None
            best_bbox = None
            for det_id, bbox in same_class_others:
                iou = self._iou(csrt_bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_id = det_id
                    best_bbox = bbox
            if best_bbox is not None and best_iou >= self.track_iou_handoff_threshold:
                self.get_logger().info(
                    f'Track-id handoff {self.goal_track_id} -> {best_det_id} '
                    f'(IoU={best_iou:.2f}, class={self.goal_class_name})'
                )
                self.goal_track_id = int(best_det_id)
                self._reseed_csrt(frame, best_bbox)
                self.yolo_match_last_t = now
                self.class_mismatch_since = None
                self.visually_tracked = True
                self._tracked_bbox_source = 'yolo'
                # Re-latch the target_info so dashboards follow the new id.
                self._publish_target_info(
                    self.goal_class_name, self.goal_track_id,
                    ApproachObject.Goal.TARGET_TOY, True,
                )
                return best_bbox

        # Class-change watchdog: detections present, none of our class.
        if saw_any_class and not saw_target_class:
            if self.class_mismatch_since is None:
                self.class_mismatch_since = now
                self.get_logger().info(
                    f'Class mismatch started (expected {self.goal_class_name}, '
                    f'seeing {list(msg.class_name)})'
                )

        self.visually_tracked = False
        # csrt_bbox may still be valid (CSRT coasting through a YOLO gap);
        # it may also be None when CSRT has lost. The caller decides what to
        # publish based on whether csrt_bbox is None.
        self._tracked_bbox_source = 'csrt' if csrt_bbox is not None else ''
        return csrt_bbox

    def _check_fusion_timeouts(self, now: float) -> Optional[str]:
        if self.goal_target_type != ApproachObject.Goal.TARGET_TOY:
            return None

        # Mirror the ArUco heuristic: only fire YOLO-based aborts when the
        # object is expected to be in view. During dead_reckoning the robot
        # is still turning toward the target — YOLO correctly sees nothing,
        # so silence is not a failure. Check both the tracking mode reported
        # by the DR node and the heading cone; if either says "not in view
        # yet", suppress the abort.
        object_should_be_visible = (
            self.tracking_mode == 'visual'
            and abs(self.last_heading_deg) <= self.reacquire_failure_deg
        )

        if object_should_be_visible and (
            self.yolo_match_last_t is not None
            and (now - self.yolo_match_last_t) > self.yolo_gap_abort_sec
        ):
            return (
                f'YOLO gap exceeded {self.yolo_gap_abort_sec:.1f}s '
                f'(last same-class match {now - self.yolo_match_last_t:.1f}s ago)'
            )
        if object_should_be_visible and (
            self.class_mismatch_since is not None
            and (now - self.class_mismatch_since) > self.class_change_abort_sec
        ):
            return (
                f'Class change persisted >{self.class_change_abort_sec:.1f}s '
                f'(expected class={self.goal_class_name})'
            )
        return None

    def _try_track_toy(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # Hold the lock for the entire body: execute_callback also holds self.lock
        # while calling tracker.init(), so without this we get a concurrent
        # init+update on the same OpenCV object → SIGSEGV in native code.
        with self.lock:
            if self.tracker is None:
                return None

            ok, bbox = self.tracker.update(frame)
            if ok:
                self.lost_since = None
                self._cache_template(frame, bbox)
                x, y, w, h = bbox
                return (int(x), int(y), int(w), int(h))

            # Track lost: try to re-acquire via normalized cross-correlation.
            if self.lost_since is None:
                self.lost_since = time.time()

            new_bbox = self._template_reacquire(frame)
            if new_bbox is not None:
                seed = tuple(int(v) for v in new_bbox)
                self.tracker = self._create_csrt()
                self.tracker.init(frame, seed)
                self.lost_since = None
                self._cache_template(frame, seed)
                x, y, w, h = new_bbox
                return (int(x), int(y), int(w), int(h))

            return None

    def _detect_markers_scaled(
        self, frame: np.ndarray,
    ) -> Tuple[Sequence, Optional[np.ndarray], float]:
        """Run the one canonical cv2.aruco pass for this frame.

        Returns ``(corners, ids, inv_scale)`` so the caller can both publish
        the markers (for the dashboard) and feed them back into the
        tracking pipeline without detecting twice.
        """
        scale = self.aruco_detect_scale
        if 0.0 < scale < 1.0:
            det_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            inv_scale = 1.0 / scale
        else:
            det_frame = frame
            inv_scale = 1.0
        corners, ids, _ = self.aruco_detector.detectMarkers(det_frame)
        return corners, ids, inv_scale

    def _publish_detected_markers(
        self,
        corners: Sequence,
        ids: Optional[np.ndarray],
        inv_scale: float,
        frame_w: float,
    ) -> None:
        """Publish all detected markers on WSKR/aruco_markers.

        Corners are scaled back into decoded-frame pixel space
        (multiplied by ``inv_scale``) and then width-normalized, so the
        downstream consumer uses the same "fraction of frame width"
        convention as tracked_bbox.
        """
        if self.markers_pub.get_subscription_count() == 0:
            return
        out = Float32MultiArray()
        n = 0 if ids is None else int(ids.size)
        out.layout.dim = [
            MultiArrayDimension(label='marker', size=n, stride=9 * max(n, 1)),
            MultiArrayDimension(label='field', size=9, stride=1),
        ]
        flat: list[float] = []
        if ids is not None and frame_w > 0.0:
            for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
                pts = marker_corners[0] * inv_scale
                flat.append(float(marker_id))
                for i in range(4):
                    flat.append(float(pts[i, 0]) / frame_w)
                    flat.append(float(pts[i, 1]) / frame_w)
        out.data = flat
        self.markers_pub.publish(out)

    def _pick_target_bbox_from_detection(
        self,
        corners: Sequence,
        ids: Optional[np.ndarray],
        inv_scale: float,
        frame_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Goal-side tracking pick from a pre-run detection pass.

        Same lost_since bookkeeping as the old _try_track_box; only the
        detection step itself has moved upstream so it can be shared with
        the markers publisher.
        """
        target_id = self.goal_object_id if self.goal_object_id >= 0 else self.aruco_id

        if ids is None:
            if self.lost_since is None:
                self.lost_since = time.time()
            return None

        flat_ids = ids.flatten().tolist()
        for i, marker_id in enumerate(flat_ids):
            if int(marker_id) != target_id:
                continue
            self.lost_since = None
            pts = corners[i][0] * inv_scale
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            self._tracked_bbox_source = 'aruco'
            return self._sanitize_bbox((x_min, y_min, x_max - x_min, y_max - y_min), frame_shape)

        # Other markers visible but not the target — count as lost.
        if self.lost_since is None:
            self.lost_since = time.time()
        return None

    def image_callback(self, msg: CompressedImage) -> None:
        """Per-frame vision: detect markers (always, when subscribed),
        publish them for the dashboard, and — if a goal is active — run
        the target-tracking pipeline off the same detection result."""
        has_goal = self.active_goal_handle is not None
        has_markers_sub = self.markers_pub.get_subscription_count() > 0
        if not has_goal and not has_markers_sub:
            return

        t_start = time.perf_counter()
        # IMREAD_REDUCED_COLOR_2: libjpeg-turbo decodes directly at half
        # resolution by skipping high-frequency DCT coefficients. Roughly
        # 2-4x faster than a full decode. All downstream logic uses
        # width-normalized coords so the smaller frame is fine.
        frame = cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2,
        )
        if frame is None:
            return
        t_decode = time.perf_counter()

        self.last_frame = frame
        frame_w = float(frame.shape[1]) if frame.shape[1] > 0 else 1.0

        skip_aruco = (
            has_goal and self.goal_target_type == ApproachObject.Goal.TARGET_TOY
        )
        if skip_aruco:
            det_corners, det_ids, det_inv_scale = (), None, 1.0
        else:
            det_corners, det_ids, det_inv_scale = self._detect_markers_scaled(frame)
            self._publish_detected_markers(det_corners, det_ids, det_inv_scale, frame_w)

        if not has_goal:
            return

        # Reset provenance every frame; set by the tracking call below when
        # a bbox is found. Empty string means "no bbox this frame".
        self._tracked_bbox_source = ''
        tracked_bbox = None
        if self.goal_target_type == ApproachObject.Goal.TARGET_TOY:
            csrt_bbox = self._try_track_toy(frame)
            tracked_bbox = self._fuse_yolo_with_csrt(frame, csrt_bbox)
        elif self.goal_target_type == ApproachObject.Goal.TARGET_BOX:
            tracked_bbox = self._pick_target_bbox_from_detection(
                det_corners, det_ids, det_inv_scale, frame.shape
            )
        t_detect = time.perf_counter()

        if tracked_bbox is None:
            self.frames_since_valid_track += 1
            self.visually_tracked = False
            self.tracked_bbox_pub.publish(TrackedBbox())
            self._maybe_warn_slow(t_start, t_decode, t_detect, None)
            return

        # Valid tracking frame. tracked_bbox is in frame-pixel coords; normalize
        # by the frame WIDTH so downstream consumers stay resolution-agnostic.
        self.frames_since_valid_track = 0
        self.last_tracked_bbox = tracked_bbox

        frame_w = float(frame.shape[1]) if frame.shape[1] > 0 else 1.0
        x, y, bw, bh = tracked_bbox
        x_n = x / frame_w
        y_n = y / frame_w
        w_n = bw / frame_w
        h_n = bh / frame_w

        bbox_msg = TrackedBbox()
        bbox_msg.x_norm = float(x_n)
        bbox_msg.y_norm = float(y_n)
        bbox_msg.w_norm = float(w_n)
        bbox_msg.h_norm = float(h_n)
        bbox_msg.source = self._tracked_bbox_source
        self.tracked_bbox_pub.publish(bbox_msg)

        # visually_tracked is set inside the fusion/pick calls above; only
        # force True for BOX where the pick function doesn't set it.
        if self.goal_target_type == ApproachObject.Goal.TARGET_BOX:
            self.visually_tracked = True
        center_u_n = x_n + w_n / 2.0
        center_v_n = y_n + h_n / 2.0
        self._compute_and_publish_heading(center_u_n, center_v_n)

        self._maybe_warn_slow(t_start, t_decode, t_detect, time.perf_counter())

    def _maybe_warn_slow(self, t_start, t_decode, t_detect, t_end) -> None:
        t_last = t_end if t_end is not None else t_detect
        total_ms = 1000.0 * (t_last - t_start)
        if total_ms < self.slow_frame_warn_ms:
            return
        decode_ms = 1000.0 * (t_decode - t_start)
        detect_ms = 1000.0 * (t_detect - t_decode)
        tail_ms = 1000.0 * (t_last - t_detect) if t_end is not None else 0.0
        self.get_logger().warn(
            f'image_callback slow: total={total_ms:.0f}ms '
            f'(decode={decode_ms:.0f}ms detect={detect_ms:.0f}ms publish={tail_ms:.0f}ms)'
        )

    def execute_callback(self, goal_handle):
        """Run the approach loop for one action goal until success, abort, or cancel."""
        with self.lock:
            self.active_goal_handle = goal_handle
            self.goal_target_type = goal_handle.request.target_type
            self.goal_object_id = int(goal_handle.request.object_id)
            self.goal_selected_obj = goal_handle.request.selected_obj
            self.visually_tracked = True
            self.tracker = None
            self.pending_toy_bbox = None
            self.frames_since_valid_track = 0
            self.last_tracked_bbox = None
            self.lost_since = None
            self.lost_template = None

            # Seed YOLO/CSRT fusion state from the selected_obj payload.
            # detection_ids[0] carries the ultralytics persistent track id
            # (see vision_processing_package/process_object_vision.py).
            sel = goal_handle.request.selected_obj
            self.goal_class_name = (
                str(sel.class_name[0]) if getattr(sel, 'class_name', None) else ''
            )
            try:
                self.goal_track_id = (
                    int(sel.detection_ids[0]) if sel.detection_ids else -1
                )
            except (TypeError, ValueError):
                self.goal_track_id = -1
            # Grace-init so the 1-second abort is measured from goal start.
            self.yolo_match_last_t = time.time()
            self.class_mismatch_since = None
            self.get_logger().info(
                f'Fusion init: track_id={self.goal_track_id} class={self.goal_class_name}'
            )

        # Hand control of WSKR/cmd_vel to the autopilot for the duration
        # of this goal. Autopilot drops its own state on enable so it
        # starts cleanly each episode.
        self._publish_autopilot_enable(True)

        if self.goal_target_type == ApproachObject.Goal.TARGET_BOX:
            self.yolo_enable_pub.publish(Bool(data=False))

        # Latch target_info for dashboards / downstream filters. For BOX,
        # track_id is the ArUco marker id; for TOY, the ByteTrack id.
        tinfo_track_id = (
            self.goal_track_id
            if self.goal_target_type == ApproachObject.Goal.TARGET_TOY
            else self.goal_object_id
        )
        tinfo_class = (
            self.goal_class_name
            if self.goal_target_type == ApproachObject.Goal.TARGET_TOY
            else 'box'
        )
        self._publish_target_info(
            tinfo_class, tinfo_track_id, self.goal_target_type, True,
        )

        if self.goal_target_type == ApproachObject.Goal.TARGET_TOY:
            w = int(self.goal_selected_obj.image_width) if self.goal_selected_obj.image_width else 0
            h = int(self.goal_selected_obj.image_height) if self.goal_selected_obj.image_height else 0
            if w <= 0 or h <= 0:
                if self.last_frame is not None:
                    h, w = self.last_frame.shape[:2]
            if w > 0 and h > 0:
                self.pending_toy_bbox = self._extract_bbox_from_selected_obj(
                    self.goal_selected_obj, self.goal_object_id, w, h
                )
                self.get_logger().info(
                    f'Seeded CSRT pending bbox={self.pending_toy_bbox} '
                    f'from selected_obj image={w}x{h}'
                )
            else:
                self.get_logger().warn('No image dimensions available to seed CSRT bbox')

            # Seed CSRT immediately on the freshest available frame so init and
            # first update run on the same frame content.
            wait_start = time.time()
            while self.last_frame is None and (time.time() - wait_start) < 1.0:
                time.sleep(0.02)
            if self.last_frame is not None and self.pending_toy_bbox is not None:
                fh, fw = self.last_frame.shape[:2]
                src_w = int(self.goal_selected_obj.image_width) if self.goal_selected_obj.image_width else fw
                src_h = int(self.goal_selected_obj.image_height) if self.goal_selected_obj.image_height else fh

                seed_bbox = self.pending_toy_bbox
                if src_w != fw or src_h != fh:
                    seed_bbox = self._rescale_bbox(seed_bbox, src_w, src_h, fw, fh)
                    self.get_logger().info(
                        f'Rescaled CSRT seed bbox from {src_w}x{src_h} to {fw}x{fh}: {seed_bbox}'
                    )

                padded = self._pad_bbox(seed_bbox, self.last_frame.shape, pad_frac=0.15)
                seed = self._sanitize_bbox(padded, self.last_frame.shape)

                if not self._is_valid_seed_bbox(seed, self.last_frame.shape):
                    self.get_logger().warn(
                        f'Invalid CSRT seed bbox after sanitize/pad: {seed} for frame={fw}x{fh}'
                    )
                else:
                    try:
                        with self.lock:
                            self.tracker = self._create_csrt()
                            ok = self.tracker.init(self.last_frame, seed)
                            if ok is False:
                                raise RuntimeError('CSRT init returned False')
                            self._cache_template(self.last_frame, seed)
                        self.get_logger().info(f'CSRT initialized on frozen frame with padded bbox={seed}')
                    except Exception as exc:
                        self.get_logger().error(f'CSRT init failed with seed={seed}: {exc}')
                        self.tracker = None
            else:
                self.get_logger().warn('Could not seed CSRT: no frame or no pending bbox')

        start = time.time()
        result = ApproachObject.Result()
        result.movement_success = True
        result.proximity_success = False
        result.movement_message = 'Approach timed out'

        while rclpy.ok() and self.active_goal_handle is goal_handle:
            elapsed = time.time() - start
            if elapsed > self.approach_timeout_sec:
                result.movement_success = False
                result.proximity_success = False
                result.movement_message = (
                    f'Approach timed out after {elapsed:.1f}s '
                    f'(limit {self.approach_timeout_sec:.1f}s)'
                )
                goal_handle.abort()
                break

            if goal_handle.is_cancel_requested:
                result.movement_success = False
                result.movement_message = 'Goal canceled'
                goal_handle.canceled()
                break

            feedback = ApproachObject.Feedback()
            feedback.tracking_mode = self.tracking_mode
            feedback.heading_to_target_deg = float(self.last_heading_deg)
            feedback.visually_tracked = bool(self.visually_tracked)
            feedback.whisker_lengths = self.latest_whiskers.tolist() if self.latest_whiskers is not None else []
            goal_handle.publish_feedback(feedback)

            # Reacquisition failure: DR says we should be staring at the target (inside
            # the ±reacquire_failure_deg cone) but vision has no bbox on the current frame.
            # Reacquisition-failure abort: only meaningful for TARGET_TOY.
            # CSRT can "successfully" lock onto the wrong object after drift,
            # so when DR says we should be looking at the target and vision
            # has been silent for N frames, we give up. For TARGET_BOX
            # (ArUco) detection is stateless — either the marker ID is
            # visible or not — so we rely on target_lost_timeout_sec and the
            # automatic reacquisition in dead_reckoning_node when a fresh
            # visual_obs arrives inside the reacquire cone. No early abort.
            if (
                self.goal_target_type == ApproachObject.Goal.TARGET_TOY
                and self.tracking_mode == 'dead_reckoning'
                and self.frames_since_valid_track >= self.reacquire_failure_frames
                and abs(self.last_heading_deg) <= self.reacquire_failure_deg
            ):
                result.movement_success = False
                result.proximity_success = False
                result.movement_message = (
                    f'Reacquisition failed inside ±{self.reacquire_failure_deg:.0f}° cone '
                    f'({self.frames_since_valid_track} frames without detection)'
                )
                self.get_logger().warn(
                    f'Aborting approach: {result.movement_message}'
                )
                goal_handle.abort()
                break

            # Fusion-level aborts: YOLO-gap timeout or persistent class mismatch.
            fusion_abort = self._check_fusion_timeouts(time.time())
            if fusion_abort is not None:
                result.movement_success = False
                result.proximity_success = False
                result.movement_message = fusion_abort
                self.get_logger().warn(f'Aborting approach: {fusion_abort}')
                goal_handle.abort()
                break

            # Target lost: abort only after reacquisition has failed for target_lost_timeout_sec.
            if self.lost_since is not None and (time.time() - self.lost_since) > self.target_lost_timeout_sec:
                result.movement_success = False
                result.proximity_success = False
                result.movement_message = f'Target lost: reacquisition failed for >{self.target_lost_timeout_sec:.1f}s'
                goal_handle.abort()
                break

            # Success when the target bbox reaches within proximity_success_mm.
            # Uses target_whisker_lengths (rays that stop at the first bbox pixel)
            # rather than floor whiskers, so only the tracked object triggers
            # success — not incidental floor obstacles at the same distance.
            # Requires a fresh valid track so a stale bbox can't falsely succeed.
            if (
                self.last_tracked_bbox is not None
                and self.frames_since_valid_track == 0
                and self.latest_target_whiskers is not None
                and float(np.min(self.latest_target_whiskers)) < self.proximity_success_mm
            ):
                closest_mm = float(np.min(self.latest_target_whiskers))
                # Immediate hard stop to Arduino
                self.stop_pub.publish(Empty())

                # Alignment loop: proportional control on heading until within deadband
                align_start = time.time()
                aligned = False
                period = 1.0 / max(1.0, self._align_rate_hz)
                while rclpy.ok() and (time.time() - align_start) < self._align_timeout_s:
                    if goal_handle.is_cancel_requested:
                        result.movement_success = False
                        result.movement_message = 'Goal canceled during alignment'
                        goal_handle.canceled()
                        break

                    # Choose heading source: fused heading (last_heading_deg)
                    with self.lock:
                        heading_deg = float(self.last_heading_deg)

                    # Check deadband
                    if abs(heading_deg) <= self._align_deadband_deg:
                        aligned = True
                        break

                    # Compute yaw command (kp * error_rad) and clamp
                    error_rad = math.radians(heading_deg)
                    yaw_cmd = max(-self._align_max_rate_rad_s, min(self._align_max_rate_rad_s, self._align_kp * error_rad))
                    tw = Twist()
                    tw.angular.z = float(yaw_cmd)
                    self.cmd_pub.publish(tw)

                    time.sleep(period)

                # Ensure we stop rotation
                self.cmd_pub.publish(Twist())

                if aligned:
                    result.proximity_success = True
                    result.movement_message = (
                        f'Target bbox within {closest_mm:.0f} mm '
                        f'(threshold {self.proximity_success_mm:.0f} mm)'
                    )
                    goal_handle.succeed()
                else:
                    result.proximity_success = False
                    result.movement_success = False
                    result.movement_message = f'Alignment failed or timed out after {self._align_timeout_s:.1f}s'
                    self.get_logger().warn(result.movement_message)
                    goal_handle.abort()
                break

            time.sleep(0.05)

        with self.lock:
            if self.active_goal_handle is goal_handle:
                self.active_goal_handle = None
                self.goal_target_type = None
                self.goal_object_id = -1
                self.goal_selected_obj = None
                self.tracker = None
                self.pending_toy_bbox = None
                self.lost_since = None
                self.lost_template = None

        self._publish_autopilot_enable(False)
        self._publish_target_info_inactive()
        self.yolo_enable_pub.publish(Bool(data=True))
        self.tracked_bbox_pub.publish(TrackedBbox())
        self.cmd_pub.publish(Twist())
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WSKRApproachActionServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
