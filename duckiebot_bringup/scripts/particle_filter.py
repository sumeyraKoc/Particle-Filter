#!/usr/bin/env python3

import math
import os
import random

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from particle import Particle
from resampling import effective_sample_size, normalize_weights, systematic_resample


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("particle_filter")

        # Parameters. These defaults match the current warehouse/room area.
        self.declare_parameter("num_particles", 700)
        self.declare_parameter("world_min_x", -16.0)
        self.declare_parameter("world_max_x", -9.0)
        self.declare_parameter("world_min_y", -28.0)
        self.declare_parameter("world_max_y", -19.0)
        self.declare_parameter("tag_map_path", "src/duckiebot_bringup/config/tag_map.yaml")

        # Motion noise. Increase slightly if particles collapse too early.
        self.declare_parameter("sigma_forward", 0.035)
        self.declare_parameter("sigma_lateral", 0.025)
        self.declare_parameter("sigma_theta", 0.020)

        # Sensor noise for range-bearing AR tag observations.
        self.declare_parameter("sigma_range", 0.45)
        self.declare_parameter("sigma_bearing", 0.35)
        self.declare_parameter("max_detection_range", 6.0)
        self.declare_parameter("camera_fov_rad", 1.40)  # approx 80 degrees
        self.declare_parameter("likelihood_floor", 1e-12)
        self.declare_parameter("resample_ess_ratio", 0.50)

        self.num_particles = int(self.get_parameter("num_particles").value)
        self.world_min_x = float(self.get_parameter("world_min_x").value)
        self.world_max_x = float(self.get_parameter("world_max_x").value)
        self.world_min_y = float(self.get_parameter("world_min_y").value)
        self.world_max_y = float(self.get_parameter("world_max_y").value)

        self.sigma_forward = float(self.get_parameter("sigma_forward").value)
        self.sigma_lateral = float(self.get_parameter("sigma_lateral").value)
        self.sigma_theta = float(self.get_parameter("sigma_theta").value)
        self.sigma_range = float(self.get_parameter("sigma_range").value)
        self.sigma_bearing = float(self.get_parameter("sigma_bearing").value)
        self.max_detection_range = float(self.get_parameter("max_detection_range").value)
        self.camera_fov_rad = float(self.get_parameter("camera_fov_rad").value)
        self.likelihood_floor = float(self.get_parameter("likelihood_floor").value)
        self.resample_ess_ratio = float(self.get_parameter("resample_ess_ratio").value)

        self.tags = self.load_map()
        self.particles = []
        self.last_odom = None
        self.path = Path()

        self.initialize_particles()

        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(ArucoMarkers, "/aruco_markers", self.obs_callback, 10)

        self.pose_pub = self.create_publisher(PoseStamped, "/pf_pose", 10)
        self.cloud_pub = self.create_publisher(PoseArray, "/particle_cloud", 10)
        self.path_pub = self.create_publisher(Path, "/pf_path", 10)

    def load_map(self):
        path = str(self.get_parameter("tag_map_path").value)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tag map not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        tags = data.get("tags", [])
        if len(tags) == 0:
            raise ValueError("tag_map.yaml must contain a non-empty 'tags' list")

        # Accept both [x, y] and {'x': ..., 'y': ...} formats.
        parsed = []
        for tag in tags:
            if isinstance(tag, dict):
                parsed.append((float(tag["x"]), float(tag["y"])))
            else:
                parsed.append((float(tag[0]), float(tag[1])))

        self.get_logger().info(f"Loaded {len(parsed)} AR tag positions")
        return parsed

    def initialize_particles(self):
        self.particles = []
        uniform_w = 1.0 / self.num_particles
        for _ in range(self.num_particles):
            p = Particle(
                random.uniform(self.world_min_x, self.world_max_x),
                random.uniform(self.world_min_y, self.world_max_y),
                random.uniform(-math.pi, math.pi),
            )
            p.weight = uniform_w
            self.particles.append(p)

    def odom_callback(self, msg):
        if self.last_odom is None:
            self.last_odom = msg
            self.publish_particles()
            self.publish_estimate()
            return

        curr = msg.pose.pose
        prev = self.last_odom.pose.pose

        dx_odom = curr.position.x - prev.position.x
        dy_odom = curr.position.y - prev.position.y

        q_curr = curr.orientation
        _, _, yaw = euler_from_quaternion([q_curr.x, q_curr.y, q_curr.z, q_curr.w])
        q_prev = prev.orientation
        _, _, prev_yaw = euler_from_quaternion([q_prev.x, q_prev.y, q_prev.z, q_prev.w])
        dtheta = wrap_angle(yaw - prev_yaw)

        # Convert odom-frame delta into robot-frame delta at the previous odom pose.
        local_dx = math.cos(-prev_yaw) * dx_odom - math.sin(-prev_yaw) * dy_odom
        local_dy = math.sin(-prev_yaw) * dx_odom + math.cos(-prev_yaw) * dy_odom

        for p in self.particles:
            noisy_dx = local_dx + random.gauss(0.0, self.sigma_forward)
            noisy_dy = local_dy + random.gauss(0.0, self.sigma_lateral)
            noisy_dtheta = dtheta + random.gauss(0.0, self.sigma_theta)

            p.x += math.cos(p.theta) * noisy_dx - math.sin(p.theta) * noisy_dy
            p.y += math.sin(p.theta) * noisy_dx + math.cos(p.theta) * noisy_dy
            p.theta = wrap_angle(p.theta + noisy_dtheta)

            # Optional hard bounds keep impossible particles outside the room from surviving.
            p.x = min(max(p.x, self.world_min_x), self.world_max_x)
            p.y = min(max(p.y, self.world_min_y), self.world_max_y)

        self.last_odom = msg
        self.publish_particles()
        self.publish_estimate()

    def obs_callback(self, msg):
        if len(msg.poses) == 0:
            return

        # Keep current particle weights as the prior; multiply by each observation likelihood.
        for observed_pose in msg.poses:
            cam_x = observed_pose.position.x       # right in camera frame
            cam_z = observed_pose.position.z       # forward in camera frame
            measured_range = math.hypot(cam_x, cam_z)
            measured_bearing = math.atan2(cam_x, cam_z)

            if measured_range <= 0.0 or measured_range > self.max_detection_range:
                continue

            for p in self.particles:
                likelihood = self.multi_hypothesis_likelihood(
                    p,
                    measured_range,
                    measured_bearing,
                )
                p.weight *= max(likelihood, self.likelihood_floor)

        normalize_weights(self.particles)

        # Publish the weighted estimate before resampling so the estimate reflects the posterior.
        self.publish_estimate()

        ess = effective_sample_size(self.particles)
        if ess < self.resample_ess_ratio * self.num_particles:
            self.particles = systematic_resample(self.particles)
            self.add_resampling_jitter()

        self.publish_particles()

    def multi_hypothesis_likelihood(self, particle, measured_range, measured_bearing):
        """
        Since all AR tags share the same ID, the observation likelihood is:
            p(z | x) = sum_i p(z | x, tag_i)
        This is the key project requirement. We do NOT pick the nearest tag.
        """
        total = 0.0
        for tx, ty in self.tags:
            dx = tx - particle.x
            dy = ty - particle.y
            expected_range = math.hypot(dx, dy)
            expected_bearing = wrap_angle(math.atan2(dy, dx) - particle.theta)

            # Camera can only detect tags roughly in front of it.
            if expected_range > self.max_detection_range:
                continue
            if abs(expected_bearing) > self.camera_fov_rad / 2.0:
                continue

            range_error = measured_range - expected_range
            bearing_error = wrap_angle(measured_bearing - expected_bearing)

            range_prob = math.exp(-0.5 * (range_error / self.sigma_range) ** 2)
            bearing_prob = math.exp(-0.5 * (bearing_error / self.sigma_bearing) ** 2)
            total += range_prob * bearing_prob

        return total

    def add_resampling_jitter(self):
        # Small roughening prevents particle impoverishment after resampling.
        for p in self.particles:
            p.x += random.gauss(0.0, 0.01)
            p.y += random.gauss(0.0, 0.01)
            p.theta = wrap_angle(p.theta + random.gauss(0.0, 0.005))
            p.x = min(max(p.x, self.world_min_x), self.world_max_x)
            p.y = min(max(p.y, self.world_min_y), self.world_max_y)

    def weighted_mean_pose(self):
        normalize_weights(self.particles)
        x = sum(p.x * p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles)
        sin_sum = sum(math.sin(p.theta) * p.weight for p in self.particles)
        cos_sum = sum(math.cos(p.theta) * p.weight for p in self.particles)
        theta = math.atan2(sin_sum, cos_sum)
        return x, y, theta

    def publish_particles(self):
        arr = PoseArray()
        arr.header.frame_id = "odom"
        arr.header.stamp = self.get_clock().now().to_msg()

        for p in self.particles:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            q = quaternion_from_euler(0, 0, p.theta)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            arr.poses.append(pose)

        self.cloud_pub.publish(arr)

    def publish_estimate(self):
        x, y, theta = self.weighted_mean_pose()

        pose = PoseStamped()
        pose.header.frame_id = "odom"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y

        q = quaternion_from_euler(0, 0, theta)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.pose_pub.publish(pose)

        self.path.header.frame_id = "odom"
        self.path.header.stamp = pose.header.stamp
        self.path.poses.append(pose)
        self.path_pub.publish(self.path)


def main():
    rclpy.init()
    node = ParticleFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
