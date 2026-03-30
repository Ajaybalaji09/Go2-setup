#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class BallAreaFinderNode(Node):
    def __init__(self):
        super().__init__('ball_area_finder')

        self.latest_color_frame = None

        # HSV range for yellow — same as main script
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # COLOR SUB — same topic as main script
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )

        self.timer = self.create_timer(0.05, self.process)

        self.get_logger().info('✅ Ball Area Finder Node Started — waiting for frames...')

    def color_callback(self, msg: Image):
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )
            self.latest_color_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().error(f'Color conversion error: {e}')

    def process(self):
        if self.latest_color_frame is None:
            return

        frame = self.latest_color_frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = frame.copy()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest)

            (x, y), radius = cv2.minEnclosingCircle(largest)
            center = (int(x), int(y))
            radius = int(radius)

            # Draw detection
            cv2.circle(display, center, radius, (0, 255, 255), 2)
            cv2.circle(display, center, 5, (0, 0, 255), -1)

            # Overlay area + radius on frame
            cv2.putText(display, f"Area: {contour_area:.1f} px2",
                        (center[0] - 60, center[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display, f"Radius: {radius} px",
                        (center[0] - 60, center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            # Print to terminal
            self.get_logger().info(
                f"Area: {contour_area:.1f} px²  |  Radius: {radius} px  |  Center: {center}"
            )

        else:
            cv2.putText(display, "No ball detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.get_logger().info("No ball detected", throttle_duration_sec=2)

        cv2.imshow("Area Finder - Detection", display)
        cv2.imshow("Area Finder - Mask", mask)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = BallAreaFinderNode()

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