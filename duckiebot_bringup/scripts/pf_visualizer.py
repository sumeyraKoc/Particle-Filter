#!/usr/bin/env python3

import rclpy
import math
import numpy as np
import matplotlib.pyplot as plt

from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray

from tf_transformations import euler_from_quaternion

import yaml

class PFVisualizer(Node):

    def __init__(self):

        super().__init__("pf_visualizer")
        self.odom_initialized = False

        self.odom_offset_x = -13.3
        self.odom_offset_y = -22.6
        self.initial_yaw = 3.1116

        self.odom_x = []
        self.odom_y = []

        self.pf_x = []
        self.pf_y = []

        self.particles = []

        self.robot_pose = None
        self.pf_pose = None
        self.tag_positions = []

        self.load_tag_map()

        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.create_subscription(
            PoseStamped,
            "/pf_pose",
            self.pf_callback,
            10
        )

        self.create_subscription(
            PoseArray,
            "/particle_cloud",
            self.cloud_callback,
            10
        )

        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.timer = self.create_timer(
            0.1,
            self.update_plot
        )


    def load_tag_map(self):

        yaml_path = "src/duckiebot_bringup/config/tag_map.yaml"

        try:

            with open(yaml_path, "r") as file:

                data = yaml.safe_load(file)

                self.tag_positions = data["tags"]

            self.get_logger().info(
                f"{len(self.tag_positions)} marker loaded."
            )

        except Exception as e:

            self.get_logger().error(
                f"Could not load tag map: {e}"
            )

    def odom_callback(self, msg):


        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y

        if not self.odom_initialized:

            self.start_odom_x = raw_x
            self.start_odom_y = raw_y

            self.odom_initialized = True

        dx_local = raw_x - self.start_odom_x
        dy_local = raw_y - self.start_odom_y

        # rotate into world/map frame
        x_rot = (
            dx_local * math.cos(self.initial_yaw)
            - dy_local * math.sin(self.initial_yaw)
        )

        y_rot = (
            dx_local * math.sin(self.initial_yaw)
            + dy_local * math.cos(self.initial_yaw)
        )

        x = x_rot + self.odom_offset_x
        y = y_rot + self.odom_offset_y

        q = msg.pose.pose.orientation

        _, _, yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        yaw += self.initial_yaw

        # normalize angle
        yaw = math.atan2(
            math.sin(yaw),
            math.cos(yaw)
        )

        self.robot_pose = (x, y, yaw)

        self.odom_x.append(x)
        self.odom_y.append(y)

    def pf_callback(self, msg):

        x = msg.pose.position.x
        y = msg.pose.position.y

        q = msg.pose.orientation

        _, _, yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        self.pf_pose = (x, y, yaw)

        self.pf_x.append(x)
        self.pf_y.append(y)

    def cloud_callback(self, msg):

        self.particles = []

        for pose in msg.poses:

            self.particles.append(
                (
                    pose.position.x,
                    pose.position.y
                )
            )

    def draw_arrow(self, x, y, theta, color):

        dx = 0.4 * math.cos(theta)
        dy = 0.4 * math.sin(theta)

        self.ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.15,
            color=color
        )

    def update_plot(self):

        self.ax.clear()

        self.ax.set_title(
            "Particle Filter Localization"
        )

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self.ax.set_xlim(-16, -9)
        self.ax.set_ylim(-28, -19)

        self.ax.grid(True)

        # ODOM TRAJECTORY
        self.ax.plot(
            self.odom_x,
            self.odom_y,
            'b--',
            label="Odometry",
            linewidth=3
        )

        # PF TRAJECTORY
        self.ax.plot(
            self.pf_x,
            self.pf_y,
            'r-',
            label="PF Estimate",
            linewidth=2
        )

        # PARTICLES
        if len(self.particles) > 0:

            px = [p[0] for p in self.particles]
            py = [p[1] for p in self.particles]

            self.ax.scatter(
                px,
                py,
                s=10,
                alpha=0.4
            )
        # MARKERS
        if len(self.tag_positions) > 0:

            mx = [p[0] for p in self.tag_positions]
            my = [p[1] for p in self.tag_positions]

            self.ax.scatter(
                mx,
                my,
                s=120,
                c='green',
                marker='s',
                label='Markers'
            )

            # optional marker labels
            for i, (x, y) in enumerate(self.tag_positions):

                self.ax.text(
                    x + 0.05,
                    y + 0.05,
                    f"M{i}",
                    fontsize=8
                )

        # ROBOT ARROW
        if self.robot_pose is not None:

            self.draw_arrow(
                self.robot_pose[0],
                self.robot_pose[1],
                self.robot_pose[2],
                "blue"
            )

        # PF ARROW
        if self.pf_pose is not None:

            self.draw_arrow(
                self.pf_pose[0],
                self.pf_pose[1],
                self.pf_pose[2],
                "red"
            )

        self.ax.legend()

        plt.draw()
        plt.pause(0.001)


def main():

    rclpy.init()

    node = PFVisualizer()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()