#!/usr/bin/env python3

import cv2
import numpy as np
import os
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from geometry_msgs.msg import Point, TransformStamped, PoseStamped
from tf2_ros import TransformListener, Buffer, TransformBroadcaster

from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class BallDetection:
    center: Tuple[int, int]
    radius: int
    distance: Optional[float]


class YellowBallDetectorNode(Node):
    def __init__(self):
        super().__init__('yellow_ball_detector')

        # TF2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.depth_scale = 0.001  # default mm → meters

        self.detected = False

        # Detect whether a display is available
        self.has_display = (
            os.environ.get('DISPLAY', '') != '' or
            os.environ.get('WAYLAND_DISPLAY', '') != ''
        )
        if not self.has_display:
            self.get_logger().info('No display detected — running headless, publishing debug image topic')

        # Camera intrinsics
        self.camera_intrinsics = None
        self.camera_frame = 'camera_link'

        # ── YOLO setup ──────────────────────────────────────────────────────
        self.model = YOLO('yolov8n.pt')
        self.ball_class_id = 32       # COCO "sports ball"
        self.conf_threshold = 0.40
        self.min_radius = 10
        # ────────────────────────────────────────────────────────────────────

        self.get_logger().info('Waiting for camera topics...')

        # Subscriptions
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        # Publishers
        self.detection_publisher = self.create_publisher(
            Point, '/ball_detection/position', 10)
        self.distance_publisher = self.create_publisher(
            Float32, '/ball_detection/distance', 10)
        self.ball_pose_map_publisher = self.create_publisher(
            PoseStamped, '/ball_detection/pose_map', 10)
        self.debug_image_publisher = self.create_publisher(
            Image, '/ball_detection/debug_image', 10)

        self.timer = self.create_timer(0.05, self.process_detection)
        self.get_logger().info('✅ Ball Detector Node Started (YOLOv8)')

    # ── COLOR ──────────────────────────────────────────────────────────────
    def color_callback(self, msg: Image):
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1)
            self.latest_color_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().error(f'Color conversion error: {e}')

    # ── CAMERA INFO ────────────────────────────────────────────────────────
    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'width': msg.width,
                'height': msg.height
            }
            self.camera_frame = msg.header.frame_id or 'camera_link'
            self.get_logger().info(
                f"✅ Camera intrinsics loaded: {self.camera_intrinsics}")

    # ── DEPTH ──────────────────────────────────────────────────────────────
    def depth_callback(self, msg: Image):
        try:
            self.get_logger().info(
                f"Depth encoding: {msg.encoding}", throttle_duration_sec=5)
            if msg.encoding == "16UC1":
                self.latest_depth_frame = np.frombuffer(
                    msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                self.depth_scale = 0.001
            elif msg.encoding == "32FC1":
                self.latest_depth_frame = np.frombuffer(
                    msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                self.depth_scale = 1.0
            else:
                self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    # ── YOLO DETECTION ─────────────────────────────────────────────────────
    def detect_ball(self, frame) -> Tuple[Optional[Tuple[int, int]], int]:
        """
        Run YOLOv8 inference and return (center, radius) of the
        highest-confidence sports ball, or (None, 0) if none found.
        """
        results = self.model(
            frame,
            classes=[self.ball_class_id],
            conf=self.conf_threshold,
            verbose=False
        )

        best_center = None
        best_radius = 0
        best_conf = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold or conf <= best_conf:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                radius = int(((x2 - x1) + (y2 - y1)) / 4)

                if radius >= self.min_radius:
                    best_center = (cx, cy)
                    best_radius = radius
                    best_conf = conf

        return best_center, best_radius

    # ── DISTANCE ───────────────────────────────────────────────────────────
    def get_distance(self, depth_frame, center, radius) -> Optional[float]:
        if center is None or radius < self.min_radius:
            return None

        x, y = center
        roi_size = max(radius // 2, 5)

        x_min = max(0, x - roi_size)
        x_max = min(depth_frame.shape[1], x + roi_size)
        y_min = max(0, y - roi_size)
        y_max = min(depth_frame.shape[0], y + roi_size)

        roi = depth_frame[y_min:y_max, x_min:x_max]
        valid = roi[roi > 0]

        if len(valid) == 0:
            return None

        return float(np.median(valid)) * self.depth_scale

    # ── PIXEL + DEPTH → 3D ─────────────────────────────────────────────────
    def pixel_to_3d(self, pixel_x: int, pixel_y: int,
                    depth: float) -> Optional[Tuple[float, float, float]]:
        if self.camera_intrinsics is None:
            return None
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        return (x, y, depth)

    # ── TF BROADCAST ───────────────────────────────────────────────────────
    def publish_ball_tf(self, position_3d, header_stamp):
        if position_3d is None:
            return
        x, y, z = position_3d
        t = TransformStamped()
        t.header.stamp = header_stamp
        t.header.frame_id = self.camera_frame
        t.child_frame_id = 'detected_ball'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    # ── MAP LOOKUP ─────────────────────────────────────────────────────────
    def get_ball_pose_in_map(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'detected_ball', rclpy.time.Time())
            t = transform.transform.translation
            return (t.x, t.y, t.z)
        except Exception as e:
            self.get_logger().debug(f"Cannot lookup ball pose in map: {e}")
            return None

    # ── PUBLISH DEBUG IMAGE ────────────────────────────────────────────────
    def publish_debug_image(self, frame):
        try:
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame
            msg.height, msg.width = frame.shape[:2]
            msg.encoding = 'bgr8'
            msg.step = frame.shape[1] * 3
            msg.data = frame.tobytes()
            self.debug_image_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Debug image publish error: {e}')

    # ── MAIN LOOP ──────────────────────────────────────────────────────────
    def process_detection(self):
        if self.latest_color_frame is None or self.latest_depth_frame is None:
            return

        if self.latest_color_frame.shape[:2] != self.latest_depth_frame.shape[:2]:
            self.get_logger().warn("Resolution mismatch!")
            return

        center, radius = self.detect_ball(self.latest_color_frame)
        distance = self.get_distance(self.latest_depth_frame, center, radius)

        display = self.latest_color_frame.copy()

        if center is not None and radius > self.min_radius:
            cv2.circle(display, center, radius, (0, 255, 255), 2)
            cv2.circle(display, center, 5, (0, 0, 255), -1)

            if distance is not None:
                self.detected = True
                cv2.putText(display, f"{distance:.3f} m",
                            (center[0] - 50, center[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Publish pixel position
                pos = Point()
                pos.x = float(center[0])
                pos.y = float(center[1])
                pos.z = float(radius)
                self.detection_publisher.publish(pos)

                # Publish distance
                dist_msg = Float32()
                dist_msg.data = distance
                self.distance_publisher.publish(dist_msg)

                # 3D position + TF
                position_3d = self.pixel_to_3d(center[0], center[1], distance)
                if position_3d:
                    self.publish_ball_tf(
                        position_3d, self.get_clock().now().to_msg())

                    # Map pose
                    ball_pose_map = self.get_ball_pose_in_map()
                    if ball_pose_map:
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id = 'map'
                        pose_msg.pose.position.x = ball_pose_map[0]
                        pose_msg.pose.position.y = ball_pose_map[1]
                        pose_msg.pose.position.z = ball_pose_map[2]
                        pose_msg.pose.orientation.w = 1.0
                        self.ball_pose_map_publisher.publish(pose_msg)
                        self.get_logger().info(
                            f"Ball in map: ({ball_pose_map[0]:.3f}, "
                            f"{ball_pose_map[1]:.3f}, {ball_pose_map[2]:.3f}) m")
            else:
                cv2.putText(display, "Distance: N/A",
                            (center[0] - 50, center[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.detected = False
            cv2.putText(display, "No ball", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display or publish debug image
        if self.has_display:
            cv2.imshow("Detection", display)
            cv2.waitKey(1)
        else:
            self.publish_debug_image(display)


def main(args=None):
    rclpy.init(args=args)
    node = YellowBallDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if node.has_display:
            cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()