import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, ChannelFloat32
import numpy as np
import open3d as o3d

import slam
import pcd
import utils

# track_0 = 0.332
# track_1 = 0.347
# track_s42 = 0.467

AVG_DISTANCE = 0.347


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.subscription = self.create_subscription(
            PointCloud2, "pointcloud_topic", self.listener_callback, 10
        )

        self.pose_graph = slam.PoseGraph()
        self.odometry = np.eye(4)
        self.frame_count = None
        self.first_frame_received = None
        self.is_moving = False
        self.mapping_complete = False

        self.distances = []

        self.publisher_loc_ = self.create_publisher(ChannelFloat32, "loc_topic", 10)
        self.publisher_odo_ = self.create_publisher(ChannelFloat32, "odo_topic", 10)
        self.publisher_map_ = self.create_publisher(PointCloud2, "map_topic", 10)

    def listener_callback(self, msg):

        self.frame_count = int(msg.header.frame_id)
        if self.first_frame_received is None:
            self.first_frame_received = int(msg.header.frame_id)

        self.get_logger().info(f"Received frame {self.frame_count}")
        self.get_logger().info(f"first frame {self.first_frame_received}")

        frame = utils.convert_ros_to_o3d(self, msg)

        # ESTIMATE INITIAL MOVEMENT
        if self.frame_count > 0 and self.is_moving == False:
            self.odometry[1, 3] = AVG_DISTANCE * self.frame_count
            self.is_moving = True

        # optimize pose graph after 225m
        translation = self.odometry[:3, 3]
        self.distance = np.linalg.norm(translation)
        self.distances.append(self.distance)

        # OPTIMIZE POSE GRAPH AFTER 225M
        if sum(self.distances) % 225 < 0.5 and sum(self.distances) != 0:
            print("distance: ", sum(self.distances))
            # OPTIMIZE POSE GRAPH
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=0.5,
                edge_prune_threshold=0.25,
                reference_node=0,
            )

            o3d.pipelines.registration.global_optimization(
                self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option,
            )

        # FINISH FIRST LAP
        if (
            self.frame_count == self.first_frame_received
            and self.pose_graph.previous_frame is not None
            and self.mapping_complete is False
        ):

            self.get_logger().info(f"Mapping complete")

            self.mapping_complete = True

            # OPTIMIZE POSE GRAPH
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=0.5,
                edge_prune_threshold=0.25,
                reference_node=0,
            )

            o3d.pipelines.registration.global_optimization(
                self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option,
            )

        # PREPROCESS FRAME
        frame_processed = pcd.preprocess_frame(frame)
        frame_without_ground = pcd.remove_ground(frame_processed, 2)
        clusters = pcd.extract_clusters(frame_without_ground, 10)
        cone_coordinates = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pcd.detect_cones_by_size(clusters))
        )

        # ESTIMATE MOTION
        self.odometry = slam.estimate_odometry(
            cone_coordinates,
            self.pose_graph.previous_frame,
            self.odometry,
        )

        self.pose_graph.previous_frame = cone_coordinates

        # ASSOCIATE LANDMARKS
        correspondence_set, estimated_pose = slam.associate_landmarks(
            cone_coordinates,
            self.pose_graph.get_map(),
            self.pose_graph.current_pose @ self.odometry,
        )

        # UPDATE POSE GRAPH
        if self.mapping_complete:
            self.pose_graph.current_pose = estimated_pose

        else:
            self.update_pose_graph(cone_coordinates, correspondence_set)

        # PUBLISH CURRENT STATE
        # pose
        self.publisher_loc_.publish(
            utils.convert_matrix_to_ros(
                "current_location", self.pose_graph.current_pose
            )
        )

        # map
        map_update = self.pose_graph.get_map()
        map_update.points.append(self.pose_graph.current_pose[:3, 3])
        self.publisher_map_.publish(utils.convert_o3d_to_ros(self, map_update))

    def update_pose_graph(self, cone_coordinates, correspondence_set):

        # register loop closures
        if not np.array_equal(self.odometry, np.eye(4)):
            self.pose_graph.add_pose(self.odometry)

        known_landmarks = []
        for frame_idx, map_idx in correspondence_set:
            landmark_idx = self.pose_graph.idxs_landmarks[map_idx]

            transformation = np.eye(4)
            transformation[:3, 3] = cone_coordinates.points[frame_idx]

            # ignore rotational information for landmark edges
            information = np.eye(6)
            information[:3, :3] = 0

            # if self.pose_graph.map.colors[map_idx][1] == 0.3:
            #     color = np.asarray([1, 0, 0])
            # else:
            #     color = np.asarray([0, 0, 0])

            self.pose_graph.add_edge(
                self.pose_graph.idxs_poses[-1],
                landmark_idx,
                transformation,
                False,
                information=information,
            )

            known_landmarks.append(frame_idx)

        # add new landmarks
        new_landmarks = cone_coordinates.select_by_index(known_landmarks, invert=True)

        for i in range(len(new_landmarks.points)):
            transformation = np.eye(4)
            transformation[:3, 3] = new_landmarks.points[i]

            self.pose_graph.add_pose(
                transformation,
                landmark=True,
                # color=new_landmarks.colors[i],
            )


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
