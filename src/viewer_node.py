import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, ChannelFloat32
import open3d as o3d
import numpy as np

import utils


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__("viewer_node")
        self.subscription = self.create_subscription(
            PointCloud2, "map_topic", self.map_callback, 10
        )

        # self.subscription = self.create_subscription(
        #     ChannelFloat32, "loc_topic", self.loc_callback, 10
        # )

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=800, height=600)

        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)

        # self.location = o3d.geometry.PointCloud()
        # self.location.colors.append(np.asarray([1, 0, 0]))
        # self.vis.add_geometry(self.location)

    # def loc_callback(self, msg):
    #     self.get_logger().info(f"Receiving location update")

    #     pose = utils.convert_ros_to_matrix(msg, (4, 4))
    #     new_location = pose[:3, 3]
    #     print(new_location)
    #     self.location.points = o3d.utility.Vector3dVector(new_location)

    #     self.update_visualizer(self.location)

    def map_callback(self, msg):
        self.get_logger().info(
            f"Receiving map (frame_count: {int(msg.header.frame_id)})"
        )

        new_map = utils.convert_ros_to_o3d(self, msg)
        self.point_cloud.points = new_map.points
        self.point_cloud.colors = o3d.utility.Vector3dVector(
            np.zeros((len(self.point_cloud.points), 3))
        )
        self.point_cloud.colors[-1] = np.asarray([1, 0, 0])

        self.update_visualizer(self.point_cloud)

    def update_visualizer(self, geometry):
        self.vis.update_geometry(geometry)
        self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
