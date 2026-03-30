#!/usr/bin/env python3
"""
Nav2 Simple Commander — Waypoint Navigation
============================================
Drives a robot through a list of (x, y, yaw) waypoints using the
nav2_simple_commander Python API (ROS 2 Humble/Iron/Jazzy).

Usage
-----
    ros2 run <your_package> nav2_waypoint_follower.py

Or run directly (if the package is sourced):
    python3 nav2_waypoint_follower.py

Prerequisites
-------------
    pip install nav2-simple-commander   # or build nav2_simple_commander from source
    source /opt/ros/<distro>/setup.bash
    # Nav2 stack must already be running (e.g. via a launch file)
"""

import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import math


def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    """
    Pure-math replacement for tf_transformations.quaternion_from_euler.
    Returns (x, y, z, w) for an intrinsic ZYX (ROS convention) rotation.
    No external packages required — only the stdlib math module.
    """
    cr = math.cos(roll  / 2.0)
    sr = math.sin(roll  / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw   / 2.0)
    sy = math.sin(yaw   / 2.0)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return (x, y, z, w)


# ---------------------------------------------------------------------------
# Helper — build a PoseStamped from (x, y, yaw_degrees)
# ---------------------------------------------------------------------------
def make_pose(navigator: BasicNavigator, x: float, y: float, yaw_deg: float) -> PoseStamped:
    """
    Create a PoseStamped in the 'map' frame.

    Args:
        navigator:  BasicNavigator instance (provides the clock).
        x:          X position in metres.
        y:          Y position in metres.
        yaw_deg:    Heading in degrees (0 = +X axis, 90 = +Y axis).

    Returns:
        A fully populated PoseStamped message.
    """
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = navigator.get_clock().now().to_msg()

    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0

    # Convert yaw (Z-axis rotation) to quaternion
    yaw_rad = math.radians(yaw_deg)
    q = quaternion_from_euler(0.0, 0.0, yaw_rad)   # roll, pitch, yaw
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]

    return pose


# ---------------------------------------------------------------------------
# Waypoint definitions — edit these to match your environment
# Format: (x_metres, y_metres, yaw_degrees)
# ---------------------------------------------------------------------------
WAYPOINTS = [
    (1.22,  -0.02,   0.0),   # Waypoint 1 — move 1 m forward, face +X
    (1.58,  -0.35,  0.0),   # Waypoint 4 — return to origin, face -Y (a square loop)
    (1.92,-1.09, 300.0),
    (2.8,-1.5,300.0),
    (4.0,-1.27,90.0),
    (3.8,0.29,180.0),
    (4.0,-0.49,200.0),
    (4.0,-1.27,180.0),
    (2.8,-1.5,180.0),
    (2.24,-1.17,180.0),
    (1.38,-1.23,230.0),
    (0.0,-1.72,300.0),
    (1.24,-1.4,0.0),
    (1.55,-0.26,90.0),
    (0.0,-0.01,180.0)
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rclpy.init()

    navigator = BasicNavigator()

    # ── 1. Set the initial pose (where the robot currently is) ──────────────
    initial_pose = make_pose(navigator, x=0.0, y=0.0, yaw_deg=0.0)
    navigator.setInitialPose(initial_pose)

    # ── 2. Wait until Nav2 is fully active ──────────────────────────────────
    navigator.get_logger().info("Waiting for Nav2 to become active…")
    navigator.waitUntilNav2Active()
    navigator.get_logger().info("Nav2 is active. Starting waypoint tour.")

    # ── 3. Build the goal-pose list ─────────────────────────────────────────
    goal_poses = [
        make_pose(navigator, x, y, yaw)
        for x, y, yaw in WAYPOINTS
    ]

    # ── 4. Send all waypoints at once ────────────────────────────────────────
    navigator.followWaypoints(goal_poses)

    # ── 5. Monitor progress ──────────────────────────────────────────────────
    total = len(goal_poses)

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()

        if feedback:
            current_wp = feedback.current_waypoint
            navigator.get_logger().info(
                f"Executing waypoint {current_wp + 1} / {total}"
            )

        # Optional: cancel after a generous timeout (30 s per waypoint)
        # if Duration(seconds=30 * total) < navigator.get_clock().now() - start_time:
        #     navigator.cancelTask()
        #     break

    # ── 6. Check final result ────────────────────────────────────────────────
    result = navigator.getResult()

    if result == TaskResult.SUCCEEDED:
        navigator.get_logger().info("✅  All waypoints reached successfully!")
    elif result == TaskResult.CANCELED:
        navigator.get_logger().warning("⚠️  Navigation was canceled.")
    elif result == TaskResult.FAILED:
        navigator.get_logger().error("❌  Navigation failed.")
    else:
        navigator.get_logger().error(f"Unknown result: {result}")

    # ── 7. Clean up ──────────────────────────────────────────────────────────
    navigator.lifecycleShutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()