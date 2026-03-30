#!/usr/bin/env python3
"""
Ball Patrol Navigator
=====================
Combines ball detection with Nav2 waypoint patrol.

Architecture — three independent lanes
---------------------------------------
  LANE 1 │ Main thread — OpenCV display loop only.
  LANE 2 │ MultiThreadedExecutor (patrol_node only) — camera, AMCL, TF, publishers.
  LANE 3 │ Navigation thread — owns BasicNavigator exclusively.

State flow (one-way, no looping back):
    PATROLLING  ->  NAVIGATING_TO_STOP_POSE  ->  MISSION_COMPLETE

On ball detection:
  - Cancel patrol.
  - Build a goal = robot's current AMCL position + STOP_OFFSET metres forward
    in the robot's current heading.  The robot stops there — it does NOT drive
    all the way to the ball's map coordinates.
  - Once that goal completes (any result), set shutdown_event and exit.
"""

import math
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from geometry_msgs.msg import (
    Point, PoseStamped, PoseWithCovarianceStamped, TransformStamped
)
from tf2_ros import TransformListener, Buffer, TransformBroadcaster

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from go2_interfaces.srv import Say


# ---------------------------------------------------------------------------
# Waypoints  (x_m, y_m, yaw_deg)
# ---------------------------------------------------------------------------
WAYPOINTS = [
    ( 0.06, 0.023,   0.0),
    ( 0.98, 0.35,   0.0),
    ( 1.36, -0.19, 270.0),
    ( 1.96, -0.62, 300.0),
    ( 2.75, -0.72,  0.0),
    ( 3.71,  -1.09, 315.0),
    ( 3.35, -0.57, 90.0),
    ( 4.33, 0.67, 180.0),
    ( 3.03, -0.65, 280.0),
    ( 2.57, -0.64, 180.0),
    ( 1.49, -0.74, 180.0),
    ( 0.07, -1.34, 180.0),
    ( 0.03, -1.38,   0.0),
    ( 1.29, -0.84,  60.0),
    ( 1.52, -0.28, 90.0),
    (0.91,0.33,120.0),
    (-0.15,0.29,180.0),
    (-0.21,0.27,0.0)
]

NAV_POLL_HZ   = 10    # Hz — how fast nav thread polls isTaskComplete
DETECT_HZ     = 20    # Hz — vision timer
DISPLAY_Q_MAX = 2     # max display frames buffered
STOP_OFFSET   = 0.4   # metres forward from detection pose (tune as needed)
AUDIO_FILE    = '/home/top/go2_ws/src/go2_driver/src/dog.wav'   # ← edit this path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll  / 2), math.sin(roll  / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw   / 2), math.sin(yaw   / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def make_pose(navigator: BasicNavigator,
              x: float, y: float, yaw_deg: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp    = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    q = quaternion_from_euler(0.0, 0.0, math.radians(yaw_deg))
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    return pose


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
class State:
    PATROLLING       = 'PATROLLING'
    NAVIGATING       = 'NAVIGATING_TO_STOP_POSE'
    MISSION_COMPLETE = 'MISSION_COMPLETE'


# ---------------------------------------------------------------------------
# Lane 2 node — camera, AMCL, TF, publishers
# ---------------------------------------------------------------------------
class BallPatrolNavigator(Node):

    def __init__(self, navigator: BasicNavigator,
                 display_queue: queue.Queue,
                 shutdown_event: threading.Event):
        super().__init__('ball_patrol_navigator')

        self.navigator       = navigator
        self._display_queue  = display_queue
        self._shutdown_event = shutdown_event

        # ── Detection state  (written by Lane 2, read by Lane 3) ────────────
        self._detect_lock  = threading.Lock()
        self.detected      = False
        self._ball_map_pos = None        # (x, y, z) in map frame

        # ── Camera / sensor ──────────────────────────────────────────────────
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.depth_scale        = 0.001
        self.camera_intrinsics  = None
        self.camera_frame       = 'camera_color_optical_frame'

        # ── AMCL — store full pose so nav thread can read position + heading ─
        self.amcl_orientation = None   # geometry_msgs/Quaternion
        self.amcl_position    = None   # geometry_msgs/Point

        # ── HSV range — yellow ball ───────────────────────────────────────────
        self.lower_color = np.array([22, 100, 100])
        self.upper_color = np.array([35, 255, 255])
        # Purple alternative:
        # self.lower_color = np.array([100, 120,  50])
        # self.upper_color = np.array([130, 255, 255])

        self.kernel          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_radius      = 20     # ← raised from 5 to match ball_detector (was too permissive)
        self.min_pixel_area  = 1500    # ← ported from ball_detector: reject small contours (px²)

        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_buffer      = Buffer()
        self.tf_listener    = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(CameraInfo,
            '/camera/camera/color/camera_info', self._camera_info_cb, 10)
        self.create_subscription(Image,
            '/camera/camera/color/image_raw', self._color_cb, 10)
        self.create_subscription(Image,
            '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped,
            '/amcl_pose', self._amcl_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_position = self.create_publisher(Point,       '/ball_detection/position', 10)
        self.pub_distance = self.create_publisher(Float32,     '/ball_detection/distance', 10)
        self.pub_pose_map = self.create_publisher(PoseStamped, '/ball_detection/pose_map', 10)

        # ── Audio service client ──────────────────────────────────────────────
        self._audio_client = self.create_client(Say, '/play_audio')

        # ── Vision timer (Lane 2) ─────────────────────────────────────────────
        self.create_timer(1.0 / DETECT_HZ, self._detection_loop)

        self.get_logger().info('BallPatrolNavigator ready.')

    # ════════════════════════════════════════════════════════════════════════
    # ROS callbacks
    # ════════════════════════════════════════════════════════════════════════

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        self.amcl_orientation = msg.pose.pose.orientation
        self.amcl_position    = msg.pose.pose.position

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': float(msg.k[0]), 'fy': float(msg.k[4]),
                'cx': float(msg.k[2]), 'cy': float(msg.k[5]),
            }
            self.camera_frame = msg.header.frame_id or self.camera_frame
            self.get_logger().info(f'Camera intrinsics: {self.camera_intrinsics}')

    def _color_cb(self, msg: Image):
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1)
            self.latest_color_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().error(f'Color cb: {e}')

    def _depth_cb(self, msg: Image):
        try:
            if msg.encoding == '16UC1':
                self.latest_depth_frame = np.frombuffer(
                    msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                self.depth_scale = 0.001
            elif msg.encoding == '32FC1':
                self.latest_depth_frame = np.frombuffer(
                    msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                self.depth_scale = 1.0
            else:
                self.get_logger().error(
                    f'Unsupported depth encoding: {msg.encoding}',
                    throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f'Depth cb: {e}')

    # ════════════════════════════════════════════════════════════════════════
    # Vision helpers
    # ════════════════════════════════════════════════════════════════════════

    def _detect_ball(self, frame):
        """
        Detect ball using HSV masking + contour filtering.

        Changes vs original:
          - Added min_pixel_area (500 px²) check — ported from ball_detector.
            Contours below this threshold are rejected with a debug log,
            preventing false positives from small colour blobs.
          - Returns (None, 0, mask) when contour area is too small, consistent
            with the 'no detection' path downstream.
        """
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center, radius = None, 0

        if contours:
            largest      = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest)

            if contour_area >= self.min_pixel_area:
                (x, y), r = cv2.minEnclosingCircle(largest)
                center, radius = (int(x), int(y)), int(r)
                self.get_logger().debug(f'Contour area: {contour_area:.1f} px²')
            else:
                # Reject — too small, likely noise or a colour-similar object
                self.get_logger().debug(
                    f'Contour rejected — area {contour_area:.1f} px² '
                    f'< min {self.min_pixel_area} px²')

        return center, radius, mask

    def _get_depth(self, depth_frame, center, radius) -> Optional[float]:
        if center is None or radius < self.min_radius:
            return None
        x, y     = center
        roi_size = max(radius // 2, 5)
        roi = depth_frame[
            max(0, y - roi_size):min(depth_frame.shape[0], y + roi_size),
            max(0, x - roi_size):min(depth_frame.shape[1], x + roi_size),
        ]
        valid = roi[roi > 0]
        return float(np.median(valid) * self.depth_scale) if len(valid) else None

    def _pixel_to_3d(self, px, py, depth) -> Optional[Tuple[float, float, float]]:
        if self.camera_intrinsics is None:
            return None
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        return ((px - cx) * depth / fx, (py - cy) * depth / fy, depth)

    def _broadcast_ball_tf(self, pos3d, stamp):
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = self.camera_frame
        t.child_frame_id  = 'detected_ball'
        t.transform.translation.x = pos3d[0]
        t.transform.translation.y = pos3d[1]
        t.transform.translation.z = pos3d[2]
        t.transform.rotation.w    = 1.0
        self.tf_broadcaster.sendTransform(t)

    def _lookup_ball_in_map(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'detected_ball', rclpy.time.Time())
            return (tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z)
        except Exception as e:
            self.get_logger().debug(f'TF lookup: {e}')
            return None

    # ════════════════════════════════════════════════════════════════════════
    # Detection loop  (Lane 2 timer — does NOT call imshow)
    # ════════════════════════════════════════════════════════════════════════

    def _detection_loop(self):
        if self.latest_color_frame is None or self.latest_depth_frame is None:
            return
        if self.latest_color_frame.shape[:2] != self.latest_depth_frame.shape[:2]:
            return

        color  = self.latest_color_frame.copy()
        depth  = self.latest_depth_frame
        center, radius, mask = self._detect_ball(color)
        distance = self._get_depth(depth, center, radius)
        display  = color.copy()

        # ball_found now requires: valid center + radius > min_radius (20)
        # + positive distance.  The min_pixel_area check inside _detect_ball
        # already gates center/radius so they are None/0 for small blobs.
        ball_found = (center is not None and radius > self.min_radius
                      and distance is not None)

        if ball_found:
            cv2.circle(display, center, radius, (0, 255, 255), 2)
            cv2.circle(display, center, 5,      (0, 0, 255),   -1)
            cv2.putText(display, f'{distance:.3f} m',
                        (center[0] - 50, center[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.pub_position.publish(Point(
                x=float(center[0]), y=float(center[1]), z=float(radius)))
            self.pub_distance.publish(Float32(data=distance))

            pos3d = self._pixel_to_3d(center[0], center[1], distance)
            if pos3d:
                stamp = self.get_clock().now().to_msg()
                self._broadcast_ball_tf(pos3d, stamp)
                ball_map = self._lookup_ball_in_map()
                if ball_map:
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp    = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = 'map'
                    pose_msg.pose.position.x = ball_map[0]
                    pose_msg.pose.position.y = ball_map[1]
                    pose_msg.pose.position.z = ball_map[2]
                    pose_msg.pose.orientation.w = 1.0
                    self.pub_pose_map.publish(pose_msg)
                    self.get_logger().info(
                        f'Ball in map: ({ball_map[0]:.3f}, {ball_map[1]:.3f}, '
                        f'{ball_map[2]:.3f}) m',
                        throttle_duration_sec=1.0)
                    with self._detect_lock:
                        self.detected      = True
                        self._ball_map_pos = ball_map
                    self._push_display(display, mask)
                    return
        else:
            label = 'No ball' if (center is None or radius <= self.min_radius) \
                               else 'Distance N/A'
            cv2.putText(display, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        with self._detect_lock:
            self.detected      = False
            self._ball_map_pos = None

        self._push_display(display, mask)

    def _push_display(self, display, mask):
        """Non-blocking push; drop oldest frame if queue is full."""
        if self._display_queue.full():
            try:
                self._display_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._display_queue.put_nowait((display, mask))
        except queue.Full:
            pass

    def get_detection(self):
        """Thread-safe snapshot — called from Lane 3."""
        with self._detect_lock:
            return self.detected, self._ball_map_pos

    def _play_audio(self, wav_path: str, timeout_sec: float = 10.0) -> bool:
        """
        Call /play_audio from the navigation thread (Lane 3).

        The service client lives on patrol_node which is being spun by the
        MultiThreadedExecutor in Lane 2 — so the response future is resolved
        automatically by that executor.  We just poll future.done() here
        without calling any extra spin.
        """
        if not self._audio_client.service_is_ready():
            self.get_logger().warn('/play_audio not ready — waiting up to 5 s...')
            if not self._audio_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('/play_audio unavailable — skipping audio.')
                return False

        req      = Say.Request()
        req.text = wav_path
        future   = self._audio_client.call_async(req)

        self.get_logger().info(f'Calling /play_audio: {wav_path}')
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error('/play_audio call timed out.')
                return False
            time.sleep(0.05)

        self.get_logger().info('/play_audio completed.')
        return True

    # ════════════════════════════════════════════════════════════════════════
    # Lane 3 — navigation thread
    # ════════════════════════════════════════════════════════════════════════

    def run_navigation(self):
        """
        Blocking loop — dedicated thread, never in the executor.

        State flow (one-way):
            PATROLLING  ->  NAVIGATING  ->  MISSION_COMPLETE  ->  exit

        When the ball is detected:
          1. Cancel patrol — robot brakes in place.
          2. Compute stop goal = current AMCL pose shifted STOP_OFFSET metres
             forward along the robot's current heading.
          3. Send that goal via navigator.goToPose() — Nav2 handles path
             planning and obstacle avoidance to reach it.
          4. Once complete (any outcome), set shutdown_event.

        The ball's map coordinates are intentionally NOT used as the
        navigation goal — the robot stops near where it currently is,
        not at the ball's physical location.
        """
        sleep_sec = 1.0 / NAV_POLL_HZ
        state     = State.PATROLLING
        self.get_logger().info('Navigation thread started.')

        while not self._shutdown_event.is_set() and rclpy.ok():

            # ── PATROLLING ───────────────────────────────────────────────────
            if state == State.PATROLLING:
                detected, ball_map = self.get_detection()

                if detected and ball_map is not None:
                    self.get_logger().info(
                        f'Ball seen at map ({ball_map[0]:.3f}, {ball_map[1]:.3f}) '
                        f'— cancelling patrol.')
                    try:
                        self.navigator.cancelTask()
                        deadline = time.time() + 2.0
                        while not self.navigator.isTaskComplete():
                            if time.time() > deadline:
                                self.get_logger().warn('Cancel timed out — proceeding.')
                                break
                            time.sleep(0.05)
                    except Exception as e:
                        self.get_logger().warn(f'cancelTask error: {e}')

                    new_state = self._go_to_stop_pose(ball_map)
                    if new_state == State.PATROLLING:
                        # AMCL not ready yet — keep patrolling and retry
                        pass
                    else:
                        state = new_state

                elif self.navigator.isTaskComplete():
                    self.get_logger().info('Starting patrol loop.')
                    goal_poses = [
                        make_pose(self.navigator, x, y, yaw)
                        for x, y, yaw in WAYPOINTS
                    ]
                    self.navigator.followWaypoints(goal_poses)

                else:
                    feedback = self.navigator.getFeedback()
                    if feedback:
                        self.get_logger().info(
                            f'Patrol wp {feedback.current_waypoint + 1}'
                            f'/{len(WAYPOINTS)}',
                            throttle_duration_sec=2.0)

            # ── NAVIGATING ───────────────────────────────────────────────────
            # Re-detections here are intentionally ignored — goal is already sent.
            elif state == State.NAVIGATING:
                if self.navigator.isTaskComplete():
                    result = self.navigator.getResult()
                    if result == TaskResult.SUCCEEDED:
                        self.get_logger().info('*** Stop pose reached — mission complete. ***')
                    elif result == TaskResult.FAILED:
                        self.get_logger().error('*** Navigation failed — stopping anyway. ***')
                    else:
                        self.get_logger().warning('*** Navigation cancelled — stopping. ***')

                    # Play audio alert — blocks until service responds or times out
                    self._play_audio(AUDIO_FILE)

                    state = State.MISSION_COMPLETE
                    self._shutdown_event.set()

            # ── MISSION_COMPLETE — shutdown_event already set, exit ───────────
            elif state == State.MISSION_COMPLETE:
                break

            time.sleep(sleep_sec)

        self.get_logger().info('Navigation thread exited.')

    def _go_to_stop_pose(self, ball_map) -> str:
        """
        Build a Nav2 goal from the robot's current AMCL pose + STOP_OFFSET
        metres forward in its current heading direction.

        Returns the new state string (NAVIGATING on success, PATROLLING if
        AMCL pose is not yet available so the caller can retry).
        """
        if self.amcl_position is None or self.amcl_orientation is None:
            self.get_logger().warn('AMCL pose not yet available — will retry.')
            return State.PATROLLING

        # Extract yaw from AMCL quaternion  (planar: only z/w matter)
        qz  = self.amcl_orientation.z
        qw  = self.amcl_orientation.w
        yaw = 2.0 * math.atan2(qz, qw)

        # Target = current position shifted forward along current heading
        goal_x = self.amcl_position.x + STOP_OFFSET * math.cos(yaw)
        goal_y = self.amcl_position.y + STOP_OFFSET * math.sin(yaw)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp    = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = 0.0
        goal.pose.orientation = self.amcl_orientation   # keep current heading

        self.get_logger().info(
            f'Ball at map ({ball_map[0]:.3f}, {ball_map[1]:.3f}) | '
            f'Robot at ({self.amcl_position.x:.3f}, {self.amcl_position.y:.3f}) '
            f'yaw={math.degrees(yaw):.1f} deg | '
            f'Stop goal: ({goal_x:.3f}, {goal_y:.3f}) '
            f'[+{STOP_OFFSET} m forward via Nav2]')

        self.navigator.goToPose(goal)
        return State.NAVIGATING


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    # BasicNavigator — Lane 3 only, never added to any executor
    navigator = BasicNavigator()

    # setInitialPose before waitUntilNav2Active so AMCL can converge
    init_pose = make_pose(navigator, 0.0, 0.0, 0.0)
    navigator.setInitialPose(init_pose)

    navigator.get_logger().info('Waiting for Nav2...')
    navigator.waitUntilNav2Active()   # blocks; uses its own internal spin
    navigator.get_logger().info('Nav2 active.')

    display_queue  = queue.Queue(maxsize=DISPLAY_Q_MAX)
    shutdown_event = threading.Event()

    patrol_node = BallPatrolNavigator(navigator, display_queue, shutdown_event)

    # Lane 2 executor — patrol_node ONLY, no navigator
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(patrol_node)

    # Lane 3 thread
    nav_thread = threading.Thread(
        target=patrol_node.run_navigation, daemon=True, name='nav_thread')
    nav_thread.start()

    # Lane 2 executor in background — frees main thread for OpenCV
    executor_thread = threading.Thread(
        target=executor.spin, daemon=True, name='executor_thread')
    executor_thread.start()

    # Lane 1 — main thread: OpenCV display
    try:
        while rclpy.ok() and not shutdown_event.is_set():
            try:
                display, mask = display_queue.get(timeout=0.05)
                cv2.imshow('Detection', display)
                cv2.imshow('Mask',      mask)
            except queue.Empty:
                pass
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass

    # Teardown
    shutdown_event.set()
    nav_thread.join(timeout=15.0)   # extra time for audio to finish playing

    try:
        navigator.cancelTask()
    except Exception:
        pass
    try:
        navigator.lifecycleShutdown()
    except Exception:
        pass

    executor.shutdown(timeout_sec=2.0)
    patrol_node.destroy_node()
    navigator.destroy_node()
    cv2.destroyAllWindows()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()