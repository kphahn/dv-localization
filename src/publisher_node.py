import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from argparse import ArgumentParser

import pcd
import fake_scenes
import utils


class PointCloudPublisher(Node):
    def __init__(self, dataset, divider=1):
        super().__init__("publisher_node")
        self.publisher_ = self.create_publisher(PointCloud2, "pointcloud_topic", 10)
        self.timer = self.create_timer(0.2, self.publish_pointcloud)
        self.dataset = dataset
        self.divider = divider
        self.frame_count = 0

    def publish_pointcloud(self):

        if self.frame_count % self.divider == 0:
            frame = self.dataset[self.frame_count]
            frame = pcd.preprocess_frame(frame)

            pointcloud_msg = utils.convert_o3d_to_ros(self, frame)
            self.publisher_.publish(pointcloud_msg)
            self.get_logger().info(f"Publishing frame {self.frame_count}.")

        self.frame_count += 1
        if self.frame_count >= len(self.dataset):
            self.frame_count = 0


def main(args=None):
    # Initialization
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--noise",
        default=False,
        help="Use frames with noise",
    )
    parser.add_argument(
        "-d",
        "--divider",
        type=int,
        default=1,
        help="Only publish every nth frame",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to track dataset",
    )
    node_args = parser.parse_args()

    dataset = fake_scenes.load_relative_frames(
        node_args.dataset_path, noise=node_args.noise
    )

    args = None
    rclpy.init(args=args)
    node = PointCloudPublisher(dataset, node_args.divider)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
