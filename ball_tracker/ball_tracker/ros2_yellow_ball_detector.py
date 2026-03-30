#!/usr/bin/env python3

import cv2
import numpy as np
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from geometry_msgs.msg import Point, TransformStamped, PoseStamped
from tf2_ros import TransformListener, Buffer, TransformBroadcaster

from dataclasses import dataclass


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
        self.depth_scale = 0.001  # default

        #-------------------------------
        self.detected = False  # Track if ball is currently detected
        #--------------------------------

        # Camera intrinsics (will be populated from CameraInfo)
        self.camera_intrinsics = None
        self.camera_frame = 'camera_link'

        # HSV range for yellow
        self.lower_yellow = np.array([22, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

        # HSV range for purple (uncomment to use)
        # self.lower_yellow = np.array([100, 120, 50])
        # self.upper_yellow = np.array([130, 255, 255])

        # self.lower_yellow = np.array([115, 100, 80])
        # self.upper_yellow = np.array([140, 255, 255])

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_radius = 20
        self.min_pixel_area = 500  # minimum contour area in pixels² — tune this value

        self.get_logger().info('Waiting for camera topics...')

        # CAMERA INFO SUB (to get intrinsics)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # COLOR SUB
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )

        # ALIGNED DEPTH SUB
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        # Publishers
        self.detection_publisher = self.create_publisher(Point, '/ball_detection/position', 10)
        self.distance_publisher = self.create_publisher(Float32, '/ball_detection/distance', 10)
        self.ball_pose_map_publisher = self.create_publisher(PoseStamped, '/ball_detection/pose_map', 10)

        # Faster loop → better sync
        self.timer = self.create_timer(0.05, self.process_detection)

        self.get_logger().info('✅ Yellow Ball Detector Node Started')

    # ---------------- COLOR ----------------
    def color_callback(self, msg: Image):
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )
            # ROS sends RGB, OpenCV expects BGR
            self.latest_color_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().error(f'Color conversion error: {e}')

    # ---------------- CAMERA INFO ----------------
    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics from CameraInfo message."""
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],  # focal length x
                'fy': msg.k[4],  # focal length y
                'cx': msg.k[2],  # principal point x
                'cy': msg.k[5],  # principal point y
                'width': msg.width,
                'height': msg.height
            }
            self.camera_frame = msg.header.frame_id or 'camera_link'
            self.get_logger().info(f"✅ Camera intrinsics loaded: {self.camera_intrinsics}")

    # ---------------- DEPTH ----------------
    def depth_callback(self, msg: Image):
        try:
            self.get_logger().info(f"Depth encoding: {msg.encoding}", throttle_duration_sec=5)

            if msg.encoding == "16UC1":
                self.latest_depth_frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    msg.height, msg.width
                )
                self.depth_scale = 0.001  # mm → meters

            elif msg.encoding == "32FC1":
                self.latest_depth_frame = np.frombuffer(msg.data, dtype=np.float32).reshape(
                    msg.height, msg.width
                )
                self.depth_scale = 1.0  # already meters

            else:
                self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
                return

        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    # ---------------- DETECTION ----------------
    def detect_yellow_ball(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None
        radius = 0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest)  # actual filled pixel area

            if contour_area >= self.min_pixel_area:  # only detect if area is large enough
                (x, y), radius = cv2.minEnclosingCircle(largest)
                center = (int(x), int(y))
                radius = int(radius)
                self.get_logger().debug(f"Contour area: {contour_area:.1f} px²")
            else:
                self.get_logger().debug(
                    f"Contour rejected — area {contour_area:.1f} px² < min {self.min_pixel_area} px²"
                )

        return center, radius, mask

    # ---------------- DISTANCE ----------------
    def get_distance(self, depth_frame, center, radius):
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

        distance = np.median(valid) * self.depth_scale
        return distance

    # -------- PIXEL + DEPTH TO 3D --------
    def pixel_to_3d(self, pixel_x: int, pixel_y: int, depth: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates + depth to 3D camera coordinates (in meters)."""
        if self.camera_intrinsics is None:
            return None

        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        # Unproject using camera intrinsics
        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth

        return (x, y, z)

    # -------- PUBLISH BALL TF TRANSFORM --------
    def publish_ball_tf(self, position_3d, header_stamp):
        """Publish ball position as a tf transform."""
        if position_3d is None:
            return

        x, y, z = position_3d

        t = TransformStamped()
        t.header.stamp = header_stamp
        t.header.frame_id = self.camera_frame
        t.child_frame_id = 'detected_ball'

        # Position (in meters)
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        # Orientation (identity - no rotation)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    # -------- LOOKUP BALL POSE IN MAP --------
    def get_ball_pose_in_map(self):
        """
        Lookup the ball's pose relative to map.
        Returns: (x, y, z) coordinates or None if transform not available.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'detected_ball',
                rclpy.time.Time()  # Use latest available transform
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            return (x, y, z)

        except Exception as e:
            self.get_logger().debug(f"Cannot lookup ball pose in map: {e}")
            return None

    # ---------------- MAIN LOOP ----------------
    def process_detection(self):
        if self.latest_color_frame is None or self.latest_depth_frame is None:
            return

        # Safety: ensure same resolution
        if self.latest_color_frame.shape[:2] != self.latest_depth_frame.shape[:2]:
            self.get_logger().warn("Resolution mismatch!")
            return

        center, radius, mask = self.detect_yellow_ball(self.latest_color_frame)
        distance = self.get_distance(self.latest_depth_frame, center, radius)

        display = self.latest_color_frame.copy()

        if center is not None and radius > self.min_radius:
            cv2.circle(display, center, radius, (0, 255, 255), 2)
            cv2.circle(display, center, 5, (0, 0, 255), -1)

            if distance is not None:
                self.detected = True
                text = f"{distance:.3f} m"
                cv2.putText(display, text, (center[0]-50, center[1]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Publish pixel position and distance
                pos = Point()
                pos.x = float(center[0])
                pos.y = float(center[1])
                pos.z = float(radius)
                self.detection_publisher.publish(pos)

                dist_msg = Float32()
                dist_msg.data = distance
                self.distance_publisher.publish(dist_msg)

                # Convert to 3D and publish TF
                position_3d = self.pixel_to_3d(center[0], center[1], distance)
                if position_3d:
                    self.publish_ball_tf(position_3d, self.get_clock().now().to_msg())

                    # Lookup ball pose in map and publish
                    ball_pose_map = self.get_ball_pose_in_map()
                    if ball_pose_map:
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id = 'map'
                        pose_msg.pose.position.x = ball_pose_map[0]
                        pose_msg.pose.position.y = ball_pose_map[1]
                        pose_msg.pose.position.z = ball_pose_map[2]
                        pose_msg.pose.orientation.w = 1.0  # Identity orientation

                        self.ball_pose_map_publisher.publish(pose_msg)
                        self.get_logger().info(
                            f"Ball in map: ({ball_pose_map[0]:.3f}, {ball_pose_map[1]:.3f}, {ball_pose_map[2]:.3f}) m"
                        )

            else:
                cv2.putText(display, "Distance: N/A", (center[0]-50, center[1]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(display, "No ball", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Detection", display)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YellowBallDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()