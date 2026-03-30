# Copyright (c) 2024 Intelligent Robotics Lab (URJC)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node




def generate_launch_description():

    composable_nodes = []

    composable_node = ComposableNode(
        package='go2_driver',
        plugin='go2_driver::Go2Driver',
        name='go2_driver',
        namespace='',

    )
    composable_nodes.append(composable_node)

    container = ComposableNodeContainer(
        name='go2_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen',
    )

    static_transform_lidar_cmd = Node(
    package="tf2_ros",
    executable="static_transform_publisher",
    name="lidar_static_transform",
    output="screen",
    arguments=[
        "--x", "0.0",       # forward/backward from base_link (meters)
        "--y", "0.0",       # left/right from base_link (meters)
        "--z", "0.1",       # height from base_link (meters)
        "--roll", "0.0",    # rotation around X axis (radians)
        "--pitch", "0.0",   # rotation around Y axis (radians)
        "--yaw", "0.0",     # rotation around Z axis (radians)
        "--frame-id", "base_link",
        "--child-frame-id", "laser",   # 👈 must match your LiDAR's frame
    ],
    )   
    # pointclod_to_laserscan_cmd = Node(
    #     package="pointcloud_to_laserscan",
    #     executable="pointcloud_to_laserscan_node",
    #     name="pointcloud_to_laserscan",
    #     namespace="",
    #     output="screen",
    #     remappings=[
    #         ("cloud_in", "pointcloud"),
    #         ("scan", "scan"),
    #     ],
    #     parameters=[{"target_frame": "base_link", "max_height": 0.35  }],
    # )
#     pointclod_to_laserscan_cmd = Node(
#     package="pointcloud_to_laserscan",
#     executable="pointcloud_to_laserscan_node",
#     name="pointcloud_to_laserscan",
#     output="screen",
#     remappings=[
#         ("cloud_in", "pointcloud"),
#         ("scan", "scan"),
#     ],
#     # parameters=[{
#     #     "target_frame": "base_link",
#     #     "min_height": 0.0,
#     #     "max_height": 0.35,
#     #     "range_min": 0.1,   # 👈 allow closer objects (10 cm) og
#     #     "range_max": 4.0
#     # }],
#     parameters=[{
#         'target_frame': 'base_link',
#         'max_height': 2.0,
#         'min_height': -0.2,
#         'angle_min': -3.14159,
#         'angle_max': 3.14159,
#         'angle_increment': 0.00872665,  # 0.5 degrees
#         'scan_time': 0.1,
#         'range_min': 0.1,
#         'range_max': 20.0,
#         'use_inf': True,
#                 }],
# )
    # odometry_node = Node(
    #     package='ros2_laser_scan_matcher',
    #     parameters=[{
    #             'base_frame': 'base_link',
    #             'odom_frame': 'odom',
    #             'laser_frame': 'base_link',
    #             'publish_odom': '/odom_filtered',
    #             'publish_tf': True
    #         }],
    #     executable='laser_scan_matcher',
    #     name='odometry_publisher',
    # )

    # Create and return the launch description
    # return LaunchDescription([
    #     launch.actions.DeclareLaunchArgument(name='use_sim_time', default_value='True',
    #                                     description='Flag to enable use_sim_time'),
    #     odometry_node
    # ])

    ld = LaunchDescription()
    ld.add_action(container)
    ld.add_action(static_transform_lidar_cmd)
    # ld.add_action(pointclod_to_laserscan_cmd)
    # ld.add_action(odosmetry_node)
    return ld
