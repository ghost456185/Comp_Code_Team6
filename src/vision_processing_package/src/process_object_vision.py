#!/usr/bin/env python3
"""YOLO inference service — object detection and property estimation.

Provides two operational modes:

  1. **Streaming** — runs YOLO + ByteTrack at ``publish_hz`` on every new
     camera frame and publishes ``ImgDetectionData`` on
     ``vision/yolo/detections``. Track IDs persist across frames so the
     approach server can follow a single object.
  2. **On-demand** — the ``detect_objects_service_v2`` service runs a single
     inference (no tracker) and returns detections for one frame.
     ``get_obj_properties_service`` extends this with a rotated re-inference
     for signed aspect-ratio estimation and a ``bbox_to_xyz`` 3-D lookup.

Topics / services:
    subscribes  camera1/image_raw/compressed
    publishes   vision/yolo/detections          (streaming)
    serves      detect_objects_service_v2        (DetectObjectsV2)
    serves      get_obj_properties_service       (GetObjProperties)
"""

import json
import math
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
import torch

from robot_interfaces.msg import ImgDetectionData
from robot_interfaces.srv import BboxToXYZ, DetectObjectsV2, GetObjProperties

from system_manager_package.constants import (
    YOLO_BBOX_TIMEOUT_SEC,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_GPU_DEVICE,
    YOLO_INPUT_SIZE,
    YOLO_MODEL_PATH,
    YOLO_PUBLISH_HZ,
    YOLO_ROTATION_TIMEOUT_SEC,
    YOLO_SIGNED_AR_ROTATION_DEG,
    YOLO_TRACKER_YAML,
)


class VisionInferenceService(Node):

    CAMERA_TOPIC = 'camera1/image_raw/compressed'
    DETECT_SERVICE_NAME = 'detect_objects_service_v2'
    OBJ_PROPERTIES_SERVICE_NAME = 'get_obj_properties_service'
    DETECTIONS_TOPIC = 'vision/yolo/detections'
    SIGNED_AR_ROTATION_DEGREES = YOLO_SIGNED_AR_ROTATION_DEG

    def __init__(self):
        super().__init__('vision_inference_service')

        self.bridge = CvBridge()
        self.latest_frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_seq = 0
        self.last_inference_seq = 0
        cb_group = ReentrantCallbackGroup()

        self.declare_parameter('confidence_threshold', YOLO_CONFIDENCE_THRESHOLD)
        self.declare_parameter('rotation_timeout_sec', YOLO_ROTATION_TIMEOUT_SEC)
        self.declare_parameter('bbox_timeout_sec', YOLO_BBOX_TIMEOUT_SEC)
        self.declare_parameter('gpu_device', YOLO_GPU_DEVICE)
        self.declare_parameter('publish_hz', YOLO_PUBLISH_HZ)
        self.declare_parameter('tracker_yaml', YOLO_TRACKER_YAML)
        self.declare_parameter('yolo_model_path', YOLO_MODEL_PATH)

        self.conf_threshold = float(self.get_parameter('confidence_threshold').value)
        self.rotation_timeout_sec = float(self.get_parameter('rotation_timeout_sec').value)
        self.bbox_timeout_sec = float(self.get_parameter('bbox_timeout_sec').value)
        self.gpu_device = int(self.get_parameter('gpu_device').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)
        self.tracker_yaml = str(self.get_parameter('tracker_yaml').value).strip()

        self.configure_cuda_device()

        model_path_param = str(self.get_parameter('yolo_model_path').value).strip()
        if model_path_param:
            model_path = Path(model_path_param)
        else:
            package_root = Path(__file__).resolve().parents[1]
            engine_path = package_root / 'models' / 'your_vision_model_here.engine'
            pt_path = package_root / 'models' / 'your_vision_model_here.pt'
            model_path = engine_path if engine_path.exists() else pt_path

        if not model_path.exists():
            raise FileNotFoundError(f'YOLO model not found at: {model_path}')

        from ultralytics import YOLO

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.yolo_uses_engine = (model_path.suffix.lower() == '.engine')
        self.model = YOLO(str(model_path))
        if self.yolo_uses_engine:
            self.get_logger().info('YOLO TensorRT engine backend enabled.')
            self.log_engine_metadata(model_path)
        else:
            self.model = self.model.to(f'cuda:{self.gpu_device}')

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CompressedImage,
            self.CAMERA_TOPIC,
            self.image_callback,
            image_qos,
        )

        self.create_service(
            DetectObjectsV2,
            self.DETECT_SERVICE_NAME,
            self.handle_detect_objects,
            callback_group=cb_group,
        )
        self.create_service(
            GetObjProperties,
            self.OBJ_PROPERTIES_SERVICE_NAME,
            self.handle_get_obj_properties,
            callback_group=cb_group,
        )

        self.detect_client = self.create_client(
            DetectObjectsV2,
            self.DETECT_SERVICE_NAME,
            callback_group=cb_group,
        )
        self.bbox_xyz_client = self.create_client(
            BboxToXYZ,
            'bbox_to_xyz_service',
            callback_group=cb_group,
        )

        # Streaming YOLO detections with persistent track IDs. BEST_EFFORT depth=1
        # so late consumers always see the freshest frame, never a backlog.
        detections_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.detections_pub = self.create_publisher(
            ImgDetectionData,
            self.DETECTIONS_TOPIC,
            detections_qos,
        )
        self.last_published_seq = 0
        if self.publish_hz <= 0.0:
            raise ValueError(f'publish_hz must be > 0 (got {self.publish_hz}).')
        period = 1.0 / self.publish_hz
        self.stream_timer = self.create_timer(
            period,
            self.publish_streaming_detections,
            callback_group=cb_group,
        )

        self._yolo_streaming_enabled = True
        self.create_subscription(
            Bool, 'WSKR/yolo_streaming_enable',
            self._on_yolo_streaming_enable, 10, callback_group=cb_group,
        )

        self.get_logger().info(
            f'VisionInferenceService initialized: model={model_path}, '
            f'conf={self.conf_threshold:.2f}, '
            f'streaming at {self.publish_hz:.1f} Hz on {self.DETECTIONS_TOPIC} '
            f'(tracker={self.tracker_yaml}).'
        )

    def configure_cuda_device(self):
        if not torch.cuda.is_available():
            self.get_logger().warn('CUDA is not available; GPU inference may fail or fall back to CPU.')
            return

        try:
            torch.cuda.set_device(self.gpu_device)
            props = torch.cuda.get_device_properties(self.gpu_device)
            self.get_logger().info(
                f'Runtime CUDA device set to cuda:{self.gpu_device} '
                f'({torch.cuda.get_device_name(self.gpu_device)}, cc={props.major}.{props.minor})'
            )
        except Exception as exc:
            self.get_logger().error(
                f'Failed to set CUDA device cuda:{self.gpu_device}: {exc}. '
                'Check the gpu_device launch parameter and available GPUs.'
            )

    def log_engine_metadata(self, engine_path: Path):
        metadata_path = Path(f'{engine_path}.meta.json')
        if not metadata_path.exists():
            self.get_logger().warn(
                f'Engine metadata file not found: {metadata_path}. '
                'Re-export with pt_to_engine.py to capture build GPU details.'
            )
            return

        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except Exception as exc:
            self.get_logger().warn(f'Could not parse engine metadata {metadata_path}: {exc}')
            return

        build_gpu = metadata.get('gpu', {})
        build_name = str(build_gpu.get('name', 'unknown'))
        build_index = build_gpu.get('index', 'unknown')
        build_cc = str(build_gpu.get('capability', 'unknown'))

        runtime_name = 'unknown'
        runtime_cc = 'unknown'
        if torch.cuda.is_available():
            try:
                runtime_name = torch.cuda.get_device_name(self.gpu_device)
                runtime_props = torch.cuda.get_device_properties(self.gpu_device)
                runtime_cc = f'{runtime_props.major}.{runtime_props.minor}'
            except Exception:
                pass

        self.get_logger().info(
            f'Engine build GPU: index={build_index}, name={build_name}, cc={build_cc}. '
            f'Runtime GPU: index={self.gpu_device}, name={runtime_name}, cc={runtime_cc}.'
        )

        if build_name != 'unknown' and runtime_name != 'unknown' and build_name != runtime_name:
            self.get_logger().warn(
                'TensorRT engine GPU model does not match runtime GPU model. '
                'Rebuild the .engine on this runtime device to avoid compatibility warnings/errors.'
            )

    def _on_yolo_streaming_enable(self, msg: Bool) -> None:
        self._yolo_streaming_enabled = msg.data

    def _preprocess_for_yolo(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = YOLO_INPUT_SIZE / h
        new_w = int(w * scale)
        resized = cv2.resize(frame, (new_w, YOLO_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        if new_w > YOLO_INPUT_SIZE:
            x0 = (new_w - YOLO_INPUT_SIZE) // 2
            return resized[:, x0 : x0 + YOLO_INPUT_SIZE]
        if new_w < YOLO_INPUT_SIZE:
            pad = YOLO_INPUT_SIZE - new_w
            return cv2.copyMakeBorder(
                resized, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=0,
            )
        return resized

    def _yolo_box_to_original(
        self,
        cx: float, cy: float, w: float, h: float,
        orig_h: int, orig_w: int,
    ) -> tuple[float, float, float, float]:
        """Invert _preprocess_for_yolo so boxes are in original camera pixel space."""
        scale = YOLO_INPUT_SIZE / orig_h
        new_w = int(orig_w * scale)
        if new_w >= YOLO_INPUT_SIZE:
            x0 = (new_w - YOLO_INPUT_SIZE) // 2
        else:
            x0 = -((YOLO_INPUT_SIZE - new_w) // 2)
        return (cx + x0) / scale, cy / scale, w / scale, h / scale

    def image_callback(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )
            with self.latest_frame_lock:
                self.latest_frame = frame
                self.latest_frame_seq += 1
        except Exception as exc:
            self.get_logger().error(f'Image decode failed: {exc}')

    def get_latest_frame_snapshot(self):
        with self.latest_frame_lock:
            return self.latest_frame, self.latest_frame_seq

    def handle_detect_objects(self, request, response):
        frame, frame_seq = self.get_latest_frame_snapshot()
        if frame is None:
            response.success = False
            return response

        try:
            rotation_text = (request.rotation_degrees or '').strip()
            frame = self.apply_optional_rotation(frame, rotation_text)

            response.detections = self.run_yolo_inference(frame, frame_seq)

            response.success = True
        except Exception as exc:
            self.get_logger().error(f'DetectObjectsV2 failed: {exc}')
            response.success = False

        return response

    def handle_get_obj_properties(self, request, response):
        try:
            base_resp = self.call_detect_objects_service(request.id, '')
            rot_text = self.SIGNED_AR_ROTATION_DEGREES
            rotated_resp = self.call_detect_objects_service(request.id, rot_text)

            base_det = self.extract_detection_by_index(base_resp.detections, request.id)
            rotated_det = self.match_rotated_detection(
                rotated_resp.detections,
                base_det,
                rot_text,
                base_resp.detections.image_width,
                base_resp.detections.image_height,
            )

            raw_ar = self._aspect_ratio(base_det)
            rotated_ar = self._aspect_ratio(rotated_det)
            signed_ar = abs(raw_ar) if (rotated_ar - raw_ar) >= 0.0 else -abs(raw_ar)

            xyz_resp = self.call_bbox_to_xyz_service(
                base_det['x'],
                base_det['y'],
                base_det['width'],
                base_det['height'],
                base_resp.detections.image_width,
                base_resp.detections.image_height,
            )

            response.success = True
            response.signed_aspect_ratio = float(signed_ar)
            response.class_name = str(base_det['class_name'])
            response.x = float(base_det['x'])
            response.y = float(base_det['y'])
            response.width = float(base_det['width'])
            response.height = float(base_det['height'])
            response.x_mm = float(xyz_resp.x_mm)
            response.y_mm = float(xyz_resp.y_mm)
            response.z_mm = float(xyz_resp.z_mm)
        except Exception as exc:
            self.get_logger().error(f'GetObjProperties failed: {exc}')
            response.success = False
            response.signed_aspect_ratio = 0.0
            response.class_name = ''
            response.x = 0.0
            response.y = 0.0
            response.width = 0.0
            response.height = 0.0
            response.x_mm = 0.0
            response.y_mm = 0.0
            response.z_mm = 0.0

        return response

    def run_yolo_inference(self, image_bgr, frame_seq=None):
        orig_h, orig_w = image_bgr.shape[:2]
        start_time = time.perf_counter()
        yolo_input = self._preprocess_for_yolo(image_bgr)
        results = self.model(yolo_input, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        msg = ImgDetectionData()
        msg.image_width = orig_w
        msg.image_height = orig_h
        msg.inference_time = elapsed_ms

        if results.boxes is None:
            return msg

        boxes = results.boxes.xywh.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if float(confidence) < self.conf_threshold:
                continue

            cx, cy, width, height = self._yolo_box_to_original(
                float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                orig_h, orig_w,
            )

            msg.detection_ids.append(str(index))
            msg.x.append(cx)
            msg.y.append(cy)
            msg.width.append(width)
            msg.height.append(height)
            msg.distance.append(0.0)
            msg.class_name.append(self.model.names[int(class_id)])
            msg.confidence.append(float(confidence))
            msg.aspect_ratio.append((width / height) if height > 0.0 else 0.0)

        if frame_seq is not None:
            self.last_inference_seq = frame_seq

        return msg

    def publish_streaming_detections(self):
        if not self._yolo_streaming_enabled:
            return
        frame, frame_seq = self.get_latest_frame_snapshot()
        if frame is None:
            return
        if frame_seq == self.last_published_seq:
            # No new frame since last tick — nothing to do.
            return

        orig_h, orig_w = frame.shape[:2]
        try:
            yolo_input = self._preprocess_for_yolo(frame)
            start_time = time.perf_counter()
            results = self.model.track(
                yolo_input,
                persist=True,
                tracker=self.tracker_yaml,
                verbose=False,
            )[0]
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        except Exception as exc:
            self.get_logger().warn(f'YOLO track() failed: {exc}')
            return

        msg = ImgDetectionData()
        msg.image_width = orig_w
        msg.image_height = orig_h
        msg.inference_time = elapsed_ms

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xywh.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            # box.id is None for new/unconfirmed tracks — prefix "tmp-" so
            # downstream consumers can distinguish from confirmed track ids.
            raw_track_ids = (
                results.boxes.id.cpu().numpy().astype(int)
                if results.boxes.id is not None
                else [None] * len(boxes)
            )

            for index, (box, confidence, class_id, track_id) in enumerate(
                zip(boxes, confidences, class_ids, raw_track_ids)
            ):
                if float(confidence) < self.conf_threshold:
                    continue

                cx, cy, width, height = self._yolo_box_to_original(
                    float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                    orig_h, orig_w,
                )

                det_id = str(int(track_id)) if track_id is not None else f'tmp-{index}'
                msg.detection_ids.append(det_id)
                msg.x.append(cx)
                msg.y.append(cy)
                msg.width.append(width)
                msg.height.append(height)
                msg.distance.append(0.0)
                msg.class_name.append(self.model.names[int(class_id)])
                msg.confidence.append(float(confidence))
                msg.aspect_ratio.append((width / height) if height > 0.0 else 0.0)

        self.detections_pub.publish(msg)
        self.last_published_seq = frame_seq

    def call_detect_objects_service(self, request_id, rotation_degrees_text):
        if not self.detect_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError('DetectObjectsV2 service unavailable.')

        req = DetectObjectsV2.Request()
        req.id = int(request_id)
        req.rotation_degrees = str(rotation_degrees_text)

        future = self.detect_client.call_async(req)
        start_time = time.perf_counter()
        while not future.done():
            if (time.perf_counter() - start_time) > self.rotation_timeout_sec:
                raise RuntimeError('DetectObjectsV2 timed out.')
            time.sleep(0.01)

        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('DetectObjectsV2 call failed.')

        return result

    def call_bbox_to_xyz_service(
        self,
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height,
        image_width,
        image_height,
    ):
        if not self.bbox_xyz_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError('BboxToXYZ service unavailable.')

        req = BboxToXYZ.Request()
        req.bbox_x = float(bbox_x)
        req.bbox_y = float(bbox_y)
        req.bbox_width = float(bbox_width)
        req.bbox_height = float(bbox_height)
        req.image_width = int(image_width)
        req.image_height = int(image_height)

        future = self.bbox_xyz_client.call_async(req)
        start_time = time.perf_counter()
        while not future.done():
            if (time.perf_counter() - start_time) > self.bbox_timeout_sec:
                raise RuntimeError('BboxToXYZ timed out.')
            time.sleep(0.01)

        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('BboxToXYZ call failed.')

        return result

    def extract_detection_by_index(self, detections_msg, detection_index):
        count = min(
            len(detections_msg.x),
            len(detections_msg.y),
            len(detections_msg.width),
            len(detections_msg.height),
            len(detections_msg.class_name),
            len(detections_msg.confidence),
        )
        if count == 0:
            raise RuntimeError('No detections returned.')
        if detection_index < 0 or detection_index >= count:
            raise RuntimeError(f'Detection index {detection_index} out of range for {count}.')

        idx = int(detection_index)
        return {
            'index': idx,
            'class_name': detections_msg.class_name[idx],
            'x': float(detections_msg.x[idx]),
            'y': float(detections_msg.y[idx]),
            'width': float(detections_msg.width[idx]),
            'height': float(detections_msg.height[idx]),
        }

    def match_rotated_detection(
        self,
        detections_msg,
        base_detection,
        rotation_degrees_text,
        image_width,
        image_height,
    ):
        count = min(
            len(detections_msg.x),
            len(detections_msg.y),
            len(detections_msg.width),
            len(detections_msg.height),
            len(detections_msg.class_name),
            len(detections_msg.confidence),
        )
        if count == 0:
            raise RuntimeError('No detections were returned for rotated inference.')

        try:
            rotation_degrees = float(rotation_degrees_text)
        except ValueError as exc:
            raise RuntimeError(
                f'Invalid signed_ar_rotation_degrees value: {rotation_degrees_text}'
            ) from exc

        center_x = float(image_width) / 2.0
        center_y = float(image_height) / 2.0
        theta = math.radians(rotation_degrees)

        shifted_x = float(base_detection['x']) - center_x
        shifted_y = float(base_detection['y']) - center_y
        predicted_x = (shifted_x * math.cos(theta)) - (shifted_y * math.sin(theta)) + center_x
        predicted_y = (shifted_x * math.sin(theta)) + (shifted_y * math.cos(theta)) + center_y

        all_candidates = [
            {
                'index': int(i),
                'class_name': detections_msg.class_name[i],
                'x': float(detections_msg.x[i]),
                'y': float(detections_msg.y[i]),
                'width': float(detections_msg.width[i]),
                'height': float(detections_msg.height[i]),
            }
            for i in range(count)
        ]
        class_candidates = [
            detection
            for detection in all_candidates
            if detection['class_name'] == base_detection['class_name']
        ]
        candidates = class_candidates or all_candidates

        return min(
            candidates,
            key=lambda detection: math.hypot(
                detection['x'] - predicted_x,
                detection['y'] - predicted_y,
            ),
        )

    def apply_optional_rotation(self, image_bgr, rotation_text):
        if rotation_text == '':
            return image_bgr

        rotation_degrees = float(rotation_text)
        rotation_center = (image_bgr.shape[1] / 2.0, image_bgr.shape[0] / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_degrees, 1.0)
        return cv2.warpAffine(
            image_bgr,
            rotation_matrix,
            (image_bgr.shape[1], image_bgr.shape[0]),
        )

    def _aspect_ratio(self, detection):
        height = float(detection['height'])
        if height <= 0.0:
            raise RuntimeError('Detected bbox height must be positive.')
        return float(detection['width']) / height


def main(args=None):
    rclpy.init(args=args)
    node = VisionInferenceService()
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