import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np

import utils


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.subscription = self.create_subscription(
            PointCloud2, "map_topic", self.listener_callback, 10
        )

        # Initialize Open3D Visualizer
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=800, height=600)

    def listener_callback(self, msg):
        self.get_logger().info(
            f"Receiving map (frame_count: {int(msg.header.frame_id)})"
        )

        # Update the Open3D PointCloud object
        self.point_cloud = utils.convert_ros_to_o3d(self, msg)

        # Update the visualizer
        self.vis.clear_geometries()
        self.vis.add_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
