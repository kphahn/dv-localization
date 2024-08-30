import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

import fake_scenes
import utils


class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__("publisher_node")
        self.publisher_ = self.create_publisher(PointCloud2, "pointcloud_topic", 10)
        self.timer = self.create_timer(0.1, self.publish_pointcloud)
        self.dataset = fake_scenes.load_relative_frames(
            "/home/kphahn/University/dv-localization/Datasets/track_1"
        )
        self.frame_count = 0

    def publish_pointcloud(self):

        if self.frame_count % 2 == 0:
            frame = self.dataset[self.frame_count]

            pointcloud_msg = utils.convert_o3d_to_ros(self, frame)
            self.publisher_.publish(pointcloud_msg)
            self.get_logger().info(f"Publishing frame {self.frame_count}.")

        # if self.frame_count % 50 == 0:
        #     time.sleep(5)

        self.frame_count += 1
        if self.frame_count >= len(self.dataset):
            self.frame_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
