import open3d as o3d
import numpy as np
import copy
import time
from argparse import ArgumentParser

import fake_scenes
import utils
import pcd
import slam

### LATEST CHANGES:
# tolerance in landmark association. Cones were missing, because deviation was too large towards the end.


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
        default=True,
        help="Visualize the pose graph after all node are added",
    )
    parser.add_argument(
        "-d",
        "--dynamic",
        default=False,
        help="Visualize the pose graph dynamically (update every time a node was added)",
    )
    parser.add_argument(
        "-e",
        "--edges",
        default=True,
        help="Show every edge",
    )
    parser.add_argument(
        "-p",
        "--poses",
        default=False,
        help="Show the orientation of each pose",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to track dataset",
    )
    args = parser.parse_args()

    VISUALIZE = args.visualize
    DYNAMIC = args.dynamic
    SHOW_EDGES = args.edges
    SHOW_POSES = args.poses
    NOISE = args.noise

    DATASET_PATH = args.dataset_path

    # load dataset
    lidar_dataset, relative_labels = fake_scenes.load_relative_frames(
        DATASET_PATH, divider=1, noise=NOISE
    )
    # ground_truth_pcd, _ = fake_scenes.load_ground_truth(DATASET_PATH)
    ground_truth_coords = (
        relative_labels[0]["blue_cones"]
        + relative_labels[0]["yellow_cones"]
        + relative_labels[0]["big_cones"]
    )

    # CREATE POSE GRAPH AND PROCESS FIRST FRAME
    pose_graph = slam.PoseGraph()
    odometry = np.eye(4)

    cone_coordinates = pcd.get_cone_coordinates(
        lidar_dataset[0], 10, relative_labels[0]
    )

    for idx in range(len(cone_coordinates.points)):

        pose = np.eye(4)
        pose[:3, 3] = cone_coordinates.points[idx]

        pose_graph.add_pose(
            pose,
            landmark=True,
            color=cone_coordinates.colors[idx],
        )

    first_frame = cone_coordinates
    previous_frame = cone_coordinates

    for idx in range(1, len(lidar_dataset)):

        print(f"Processing frame: {idx}", flush=True, end="\r")

        # PREPROCESS FRAME
        cone_coordinates = pcd.get_cone_coordinates(
            lidar_dataset[idx], 10, relative_labels[idx]
        )

        # PROCESS COORDINATES
        odometry = slam.estimate_odometry(
            cone_coordinates, previous_frame, 1.5, odometry
        )
        pose_graph.add_pose(odometry)

        # ASSOCIATE LANDMARKS
        correspondence_set = slam.associate_landmarks(
            cone_coordinates, pose_graph.get_map(), pose_graph.current_pose, idx
        )

        known_landmarks = []

        # REGISTER LOOP CLOSURES
        for frame_idx, map_idx in correspondence_set:
            landmark_idx = pose_graph.idxs_landmarks[map_idx]

            transformation = np.eye(4)
            transformation[:3, 3] = cone_coordinates.points[frame_idx]

            # ignore rotational information for landmark edges
            information = np.eye(6)
            information[:3, :3] = 0

            if pose_graph.map.colors[map_idx][1] == 0.3:
                color = np.asarray([1, 0, 0])
            else:
                color = np.asarray([0, 0, 0])

            pose_graph.add_edge(
                pose_graph.idxs_poses[-1],
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

            pose_graph.add_pose(
                transformation,
                landmark=True,
                color=new_landmarks.colors[i],
            )

        previous_frame = cone_coordinates

        # TO DO:
        # - Dynamic visualization
        # - Compress landmark association into function

    # utils.draw([pose_graph.get_map()])
    pose_graph.visualize()

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.1,
        edge_prune_threshold=0.25,
        reference_node=0,
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    # utils.draw([pose_graph.get_map()])
    pose_graph.visualize()

    print("Program finished")
