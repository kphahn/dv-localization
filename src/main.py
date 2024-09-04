import open3d as o3d
import numpy as np
from argparse import ArgumentParser

import fake_scenes
import utils
import slam
import pcd

# track_0 = 0.332
# track_1 = 0.347
# track_s42 = 0.467

AVG_DISTANCE = 0.347


def update_view(vis, pcd, frame):
    pcd.points = frame.points

    vis.update_geometry(pcd)
    vis.reset_view_point(True)
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
    is_moving = False

    relevant_points = []

    mapping_in_progress = True

    for glob in range(100):

        for idx in range(0, len(lidar_dataset)):

            frame = lidar_dataset[idx]

            print(f"Processing frame: {idx}")

            # ESTIMATE INITIAL MOVEMENT
            if idx > 0 and is_moving == False:
                odometry[1, 3] = AVG_DISTANCE * idx
                is_moving = True

            # OPTIMIZE POSE GRAPH
            if (
                idx == 0
                and pose_graph.previous_frame is not None
                and mapping_in_progress is True
            ):

                print("Mapping complete")

                mapping_in_progress = False

                # OPTIMIZE POSE GRAPH
                pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
                pose.transform(pose_graph.current_pose)
                utils.draw([pose_graph.get_map(), pose_graph.get_path(), pose])

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

                pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
                pose.transform(pose_graph.current_pose)
                utils.draw([pose_graph.get_map(), pose])

            # PREPROCESS FRAME
            frame_processed = pcd.preprocess_frame(frame)
            frame_without_ground = pcd.remove_ground(frame, 0)
            clusters = pcd.extract_clusters(frame_without_ground, 10)
            cone_coordinates = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pcd.detect_cones_by_size(clusters))
            )

            # ESTIMATE MOTION
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

            pose_graph.current_pose = estimated_pose

            # UPDATE POSE GRAPH
            if mapping_in_progress:

                # register loop closure
                pose_graph.add_pose(odometry)

                known_landmarks = []
                for frame_idx, map_idx in correspondence_set:
                    landmark_idx = pose_graph.idxs_landmarks[map_idx]

                    transformation = np.eye(4)
                    transformation[:3, 3] = cone_coordinates.points[frame_idx]

                    # ignore rotational information for landmark edges
                    information = np.eye(6)
                    information[:3, :3] = 0

                    # if pose_graph.map.colors[map_idx][1] == 0.3:
                    #     color = np.asarray([1, 0, 0])
                    # else:
                    #     color = np.asarray([0, 0, 0])

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
                        # color=new_landmarks.colors[i],
                    )

            #
            #
            #
            #
            # DEBUG
            p = pose_graph.current_pose[:3, 3]
            pc = o3d.geometry.PointCloud()
            pc.points.append(p)

            update_view(vis, view, pose_graph.get_map() + pc)

    print("Program finished")
