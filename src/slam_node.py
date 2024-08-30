import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np

import slam
import utils


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.subscription = self.create_subscription(
            PointCloud2, "pointcloud_topic", self.listener_callback, 10
        )

        self.pose_graph = slam.PoseGraph()
        self.odometry = np.eye(4)
        self.frame_count = 0

        self.publisher_loc_ = self.create_publisher(PointCloud2, "location_topic", 10)
        self.publisher_map_ = self.create_publisher(PointCloud2, "map_topic", 10)

    def listener_callback(self, msg):

        self.frame_count = int(msg.header.frame_id)
        self.get_logger().info(f"Received frame {self.frame_count}")

        frame = utils.convert_ros_to_o3d(self, msg)

        # self.frame_count = idx
        # frame = msg

        self.update_pose_graph(frame)

        # self.publisher_loc_.publish(self.pose_graph.get_loc())

        # if self.frame_count % 50:

        self.pose_graph, previous_path = slam.optimize_pose_graph(self.pose_graph)

        self.publisher_map_.publish(
            utils.convert_o3d_to_ros(self, self.pose_graph.get_map())
        )

        # self.frame_count = 0

    def update_pose_graph(self, frame):

        self.get_logger().info(f"Processing...")

        # PREPROCESS FRAME
        cone_coordinates = slam.get_cone_coordinates(frame, 10)

        # REGISTER LOOP CLOSURES
        known_landmarks = []

        if self.pose_graph.previous_frame is not None:

            self.odometry = slam.estimate_odometry(
                cone_coordinates, self.pose_graph.previous_frame, 1.5, self.odometry
            )

            self.pose_graph.add_pose(self.odometry)

            # ASSOCIATE LANDMARKS
            correspondence_set = slam.associate_landmarks(
                cone_coordinates,
                self.pose_graph.get_map(),
                self.pose_graph.current_pose,
            )

            # REGISTER LOOP CLOSURES
            for frame_idx, map_idx in correspondence_set:
                landmark_idx = self.pose_graph.idxs_landmarks[map_idx]

                transformation = np.eye(4)
                transformation[:3, 3] = cone_coordinates.points[frame_idx]

                # ignore rotational information for landmark edges
                information = np.eye(6)
                information[:3, :3] = 0

                if self.pose_graph.map.colors[map_idx][1] == 0.3:
                    color = np.asarray([1, 0, 0])
                else:
                    color = np.asarray([0, 0, 0])

                self.pose_graph.add_edge(
                    self.pose_graph.idxs_poses[-1],
                    landmark_idx,
                    transformation,
                    False,
                    information=information,
                    color=color,
                )

                known_landmarks.append(frame_idx)

        # ADD NEW LANDMARKS
        new_landmarks = cone_coordinates.select_by_index(known_landmarks, invert=True)

        for i in range(len(new_landmarks.points)):
            transformation = np.eye(4)
            transformation[:3, 3] = new_landmarks.points[i]

            self.pose_graph.add_pose(
                transformation,
                landmark=True,
                color=new_landmarks.colors[i],
            )

        self.pose_graph.previous_frame = cone_coordinates


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()

    # import fake_scenes

    # dataset = fake_scenes.load_relative_frames(
    #     "/home/kphahn/University/dv-localization/Datasets/track_1"
    # )

    # for i in range(3):
    #     for idx, frame in enumerate(dataset):
    #         node.listener_callback(frame, idx)

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
