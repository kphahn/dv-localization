import open3d as o3d
import numpy as np
from argparse import ArgumentParser

import fake_scenes
import utils
import slam
import pcd

import time
import copy


def update_view(vis, pcd, frame):
    pcd.points = frame.points

    vis.update_geometry(pcd)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()
    # time.sleep(0.1)


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

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1500, height=1500)

    view = o3d.geometry.PointCloud()
    vis.add_geometry(view)

    # load dataset
    lidar_dataset = fake_scenes.load_relative_frames(
        DATASET_PATH, divider=1, noise=args.noise
    )

    # CREATE POSE GRAPH
    pose_graph = slam.PoseGraph()
    odometry = np.eye(4)

    for glob in range(10):

        for idx in range(0, len(lidar_dataset)):

            frame = lidar_dataset[idx]

            print(f"Processing frame: {idx}")
            # update_view(vis, view, frame)
            # print("original")
            # utils.draw(frame)

            # print("downsampled")
            # cp = frame.voxel_down_sample(0.05)
            # utils.draw(cp)

            # PREPROCESS FRAME
            frame_processed = pcd.preprocess_frame(frame)
            frame_without_ground = pcd.remove_ground(frame_processed, 2)
            clusters = pcd.extract_clusters(frame_without_ground, 10)
            cones = pcd.filter_clusters(clusters)
            cone_coordinates = pcd.get_cone_coordinates(cones)

            # cp = copy.deepcopy(cone_coordinates)
            # update_view(vis, view, cp)

            # PROCESS COORDINATES
            known_landmarks = []

            if pose_graph.previous_frame is not None:

                odometry = slam.estimate_odometry(
                    cone_coordinates,
                    pose_graph.previous_frame,
                    odometry,
                    glob,
                    idx,
                )

                pose_graph.add_pose(odometry)

                # ASSOCIATE LANDMARKS
                correspondence_set = slam.associate_landmarks(
                    cone_coordinates,
                    pose_graph.get_map(),
                    pose_graph.current_pose,
                    glob,
                    idx,
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

            # if idx == 253:
            # if (len(new_landmarks.points) > 1 and glob == 0) or (
            #     len(new_landmarks.points) and glob > 0
            # ):

            #     if (glob == 0 and idx == 0) or idx in [6, 81, 92, 150, 258]:
            #         pass

            #     else:

            #         for point in cone_coordinates.points:
            #             print(point)

            #         utils.draw(
            #             [
            #                 frame.paint_uniform_color(np.asarray([0, 0, 1])),
            #                 processed_frame.paint_uniform_color(np.asarray([0, 1, 0])),
            #                 pcd.remove_ground(processed_frame),
            #                 copy.deepcopy(cone_coordinates).paint_uniform_color(
            #                     np.asarray([1, 0, 0])
            #                 ),
            #             ]
            #         )

            #         odometry = slam.estimate_odometry(
            #             cone_coordinates,
            #             pose_graph.previous_frame,
            #             odometry,
            #             glob,
            #             idx,
            #             show=True,
            #         )

            #         correspondence_set = slam.associate_landmarks(
            #             cone_coordinates,
            #             pose_graph.get_map(),
            #             pose_graph.current_pose,
            #             glob,
            #             idx,
            #             show=True,
            #         )

            for i in range(len(new_landmarks.points)):
                transformation = np.eye(4)
                transformation[:3, 3] = new_landmarks.points[i]

                pose_graph.add_pose(
                    transformation,
                    landmark=True,
                    color=new_landmarks.colors[i],
                )

            update_view(vis, view, pose_graph.get_map())

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
