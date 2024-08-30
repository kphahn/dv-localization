import open3d as o3d
import numpy as np
from argparse import ArgumentParser

import fake_scenes
import utils
import slam


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
        "dataset_path",
        type=str,
        help="Path to track dataset",
    )
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path

    # load dataset
    lidar_dataset, relative_labels = fake_scenes.load_relative_frames(
        DATASET_PATH, divider=1, noise=args.noise
    )

    # CREATE POSE GRAPH
    pose_graph = slam.PoseGraph()
    odometry = np.eye(4)

    for glob in range(3):

        for idx in range(0, len(lidar_dataset)):

            print(f"Processing frame: {idx}", flush=True, end="\r")

            # PREPROCESS FRAME
            cone_coordinates = slam.get_cone_coordinates(lidar_dataset[idx], 10)

            # PROCESS COORDINATES
            known_landmarks = []

            if pose_graph.previous_frame is not None:

                odometry = slam.estimate_odometry(
                    cone_coordinates,
                    pose_graph.previous_frame,
                    1.5,
                    odometry,
                )

                pose_graph.add_pose(odometry)

                # ASSOCIATE LANDMARKS
                correspondence_set = slam.associate_landmarks(
                    cone_coordinates,
                    pose_graph.get_map(),
                    pose_graph.current_pose,
                )

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
            new_landmarks = cone_coordinates.select_by_index(
                known_landmarks, invert=True
            )

            for i in range(len(new_landmarks.points)):
                transformation = np.eye(4)
                transformation[:3, 3] = new_landmarks.points[i]

                pose_graph.add_pose(
                    transformation,
                    landmark=True,
                    color=new_landmarks.colors[i],
                )

            pose_graph.previous_frame = cone_coordinates

        # OPTIMIZE POSE GRAPH
        pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
        pose.transform(pose_graph.current_pose)
        utils.draw([pose_graph.get_map(), pose_graph.get_path(), pose])

        pose_graph, previous_path = slam.optimize_pose_graph(pose_graph)

        pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
        pose.transform(pose_graph.current_pose)
        utils.draw([pose_graph.get_map(), pose])

    print("Program finished")
