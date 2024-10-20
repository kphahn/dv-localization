import open3d as o3d
import numpy as np
from argparse import ArgumentParser

import fake_scenes
import slam
import pcd


def get_initial_distance(dataset_name):

    if dataset_name == "track_0_livox":
        return 0.29364

    elif dataset_name == "track_1_livox":
        return 0.38553

    elif dataset_name == "track_s42_livox":
        return 0.55665

    return 0


vis = o3d.visualization.Visualizer()
view = o3d.geometry.PointCloud()


def update_view(vis, pcd, frame):
    pcd.points = frame.points

    vis.update_geometry(pcd)
    # vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()


if __name__ == "__main__":

    # Initialization
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--noise",
        default=False,
        help="Use frames with noise",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        default=False,
        help="Dynamically visualize the build-up of the map",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to track dataset",
    )
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    VISUALIZATION = args.visualize

    dataset_name = DATASET_PATH.rstrip("/").rsplit("/", 1)[1]
    initial_dist = get_initial_distance(dataset_name)

    if VISUALIZATION:
        vis.create_window(width=1500, height=1500)
        vis.add_geometry(view)

    # load dataset
    lidar_dataset = fake_scenes.load_relative_frames(
        DATASET_PATH, divider=1, noise=args.noise
    )

    # CREATE POSE GRAPH
    pose_graph = slam.PoseGraph()
    is_moving = False
    mapping_complete = False
    odometry = np.eye(4)

    distances = []

    for run in range(5):
        for idx in range(0, len(lidar_dataset)):

            frame = lidar_dataset[idx]

            print(f"Processing frame: {idx}")

            # OPTIMIZE POSE GRAPH
            if (
                idx == 0
                and pose_graph.previous_frame is not None
                and mapping_complete is False
            ):

                print("Mapping complete")

                mapping_complete = True
                is_moving = False  # allow odometry adjustment for larger distance from last to first frame

                # OPTIMIZE POSE GRAPH
                option = o3d.pipelines.registration.GlobalOptimizationOption(
                    max_correspondence_distance=0.5,
                    edge_prune_threshold=0.25,
                    reference_node=0,
                )

                o3d.pipelines.registration.global_optimization(
                    pose_graph,
                    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                    option,
                )

            # PREPROCESS FRAME
            frame_processed = pcd.preprocess_frame(frame)
            frame_without_ground = pcd.remove_ground(frame, 0)
            clusters = pcd.extract_clusters(frame_without_ground, 10)
            cone_coordinates = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pcd.detect_cones_by_size(clusters))
            )

            # ESTIMATE MOTION
            # initial movment
            if idx > 0 and is_moving == False:
                odometry = np.eye(4)
                odometry[1, 3] = initial_dist * idx
                is_moving = True

            odometry = slam.estimate_odometry(
                cone_coordinates,
                pose_graph.previous_frame,
                odometry,
            )

            pose_graph.previous_frame = cone_coordinates

            # ASSOCIATE LANDMARKS
            correspondence_set, estimated_pose = slam.associate_landmarks(
                cone_coordinates,
                pose_graph.get_map(),
                pose_graph.current_pose @ odometry,
            )

            # UPDATE POSE GRAPH
            if mapping_complete:
                pose_graph.current_pose = estimated_pose

            else:

                if not np.array_equal(odometry, np.eye(4)):
                    pose_graph.add_pose(odometry)

                # register loop closure
                known_landmarks = []
                for frame_idx, map_idx in correspondence_set:
                    landmark_idx = pose_graph.idxs_landmarks[map_idx]

                    transformation = np.eye(4)
                    transformation[:3, 3] = cone_coordinates.points[frame_idx]

                    # ignore rotational information for landmark edges
                    information = np.eye(6)
                    information[:3, :3] = 0

                    pose_graph.add_edge(
                        pose_graph.idxs_poses[-1],
                        landmark_idx,
                        transformation,
                        False,
                        information=information,
                    )

                    known_landmarks.append(frame_idx)

                # add new landmarks
                new_landmarks = cone_coordinates.select_by_index(
                    known_landmarks, invert=True
                )

                for i in range(len(new_landmarks.points)):
                    transformation = np.eye(4)
                    transformation[:3, 3] = new_landmarks.points[i]

                    pose_graph.add_pose(
                        transformation,
                        landmark=True,
                    )

                # close lap
                if idx == len(lidar_dataset) - 1:
                    odometry = slam.estimate_odometry(
                        lidar_dataset[0],
                        cone_coordinates,
                        odometry,
                    )

                    pose_graph.add_edge(
                        pose_graph.idxs_poses[-1],
                        pose_graph.idxs_poses[0],
                        odometry,
                        False,
                    )

            # optimize pose graph after 225m
            translation = odometry[:3, 3]
            distance = np.linalg.norm(translation)
            distances.append(distance)
            sum_distances = sum(distances)

            if sum_distances % 225 < 0.5 and sum_distances != 0 and sum_distances < 226:
                print("distance: ", sum(distances))

                option = o3d.pipelines.registration.GlobalOptimizationOption(
                    max_correspondence_distance=0.5,
                    edge_prune_threshold=0.25,
                    reference_node=0,
                )

                o3d.pipelines.registration.global_optimization(
                    pose_graph,
                    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                    option,
                )

            # VISUALIZATION
            if VISUALIZATION:
                p = pose_graph.current_pose[:3, 3]
                pc = o3d.geometry.PointCloud()
                pc.points.append(p)

                update_view(vis, view, pose_graph.get_map() + pc)

print("Program finished")
