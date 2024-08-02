import open3d as o3d
import numpy as np
import copy
import time
from argparse import ArgumentParser

import fake_scenes
import utils
import pcd
import slam

if __name__ == "__main__":

    # Initialization
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--first",
        default=False,
        help="Only show first frame and quit",
    )
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

    if args.first:
        pcd = utils.load_dataset(DATASET_PATH, limit=1, noise=NOISE)
        utils.draw(pcd)
        exit()

    # load dataset
    lidar_dataset, relative_labels = fake_scenes.load_relative_frames(
        DATASET_PATH, divider=2
    )
    # ground_truth_pcd, ground_truth_coords = fake_scenes.load_ground_truth(DATASET_PATH)

    pose_graph = slam.PoseGraph()
    odometry = np.eye(4)

    cone_coordinates = pcd.get_cone_coordinates(
        lidar_dataset[0], 10, relative_labels[0]["big_cones"]
    )

    # for idx in range(len(cone_coordinates.points)):

    #     pose = np.eye(4)
    #     pose[:3, 3] = cone_coordinates.points[idx]

    #     pose_graph.add_pose(
    #         pose,
    #         landmark=True,
    #         color=cone_coordinates.colors[idx],
    #     )

    # utils.draw(pose_graph.get_map())

    # exit()

    previous_frame = cone_coordinates

    for idx in range(1, len(lidar_dataset)):

        print(f"Processing frame: {idx}", flush=True, end="\r")

        # PREPROCESS FRAME
        cone_coordinates = pcd.get_cone_coordinates(
            lidar_dataset[idx], 10, relative_labels[idx]["big_cones"]
        )

        # PROCESS COORDINATES
        odometry = slam.estimate_odometry(
            cone_coordinates, previous_frame, 1.5, odometry
        )
        pose_graph.add_pose(odometry)

        # ASSOCIATE LANDMARKS
        # correspondence_set = slam.associate_landmarks(
        #     cone_coordinates, pose_graph.map, pose_graph.current_pose
        # )

        # known_landmarks = []
        # loop_closure_detected = False

        # # create edges between current pose and known landmarks
        # for frame_idx, map_idx in correspondence_set:
        #     landmark_idx = pose_graph.idxs_landmarks[map_idx]

        #     transform = np.eye(4)
        #     transform[:3, 3] = cone_coordinates.points[frame_idx]

        #     pose_graph.add_edge(
        #         pose_graph.idxs_poses[-1],
        #         landmark_idx,
        #         transform,
        #         False,
        #     )

        #     if pose_graph.map.colors[map_idx][1] == 0.3:
        #         loop_closure_detected = True

        #     known_landmarks.append(frame_idx)

        # # add new landmarks
        # new_landmarks = cone_coordinates.select_by_index(known_landmarks, invert=True)
        # for idx in range(len(new_landmarks.points)):
        #     transform = np.eye(4)
        #     transform[:3, 3] = new_landmarks.points[idx]

        #     pose_graph.add_pose(
        #         transform, landmark=True, color=new_landmarks.colors[idx]
        #     )

        # # register loop closure
        # if loop_closure_detected:
        if idx == len(lidar_dataset) - 1:
            pose_graph.add_loop_closure(pose_graph.idxs_poses[-1], 0)

        previous_frame = cone_coordinates
        # TO DO:
        # - Visualize edges
        # - Dynamic visualization

    pose_graph.visualize(show_edges=True)

    # utils.draw(pose_graph.path)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=1.5,
        edge_prune_threshold=0.25,
        reference_node=0,
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    # utils.draw(path)

    # # optimize pose graph
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # option = o3d.pipelines.registration.GlobalOptimizationOption(
    #     max_correspondence_distance=1.5,
    #     edge_prune_threshold=0.25,
    #     reference_node=0,
    # )

    # o3d.pipelines.registration.global_optimization(
    #     pose_graph,
    #     o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    #     o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    #     option,
    # )

    pose_graph.visualize(show_edges=True)

    print("Program finished")

    exit()

    lidar_dataset = [pcd.remove_ground(frame) for frame in lidar_dataset]

    # lidar_dataset[0].paint_uniform_color([0, 1, 0])
    # lidar_dataset[len(lidar_dataset) - 1].paint_uniform_color([1, 0, 0])

    for frame in utils.frame_feeder(lidar_dataset):
        utils.draw(frame)

    # utils.draw(lidar_dataset)
    # exit()

    # initialize pose graph
    pose_graph = slam.PoseGraph(np.eye(4))

    ### (VISUAL) ### dynamic
    if DYNAMIC:
        pose_graph_dyn = copy.deepcopy(lidar_dataset)
        poses = o3d.geometry.TriangleMesh.create_coordinate_frame()
        edges = o3d.geometry.LineSet()
        path = o3d.geometry.PointCloud()

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.poll_events()
        vis.update_renderer()

    # run mapping loop
    idx_previous_pose = 0
    idx_current_pose = 0
    for idx, current_frame in enumerate(lidar_dataset):

        print(f"Processing frame {idx}", flush=True, end="\r")

        # update indexes
        previous_frame = lidar_dataset[idx - 1]
        idx_previous_pose = idx_current_pose
        idx_current_pose = len(pose_graph.nodes)

        # register poses
        if idx == 0:
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(relative_transform)
            )  # first frame sets global coordinate system
            node_types.append(slam.NODE_TYPE.POSE)

        else:
            icp_result = o3d.pipelines.registration.registration_icp(
                current_frame,
                previous_frame,
                1.5,
                relative_transform,
            )
            relative_transform = icp_result.transformation
            global_transform = (
                pose_graph.nodes[idx_previous_pose].pose @ relative_transform
            )

            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(global_transform)
            )
            node_types.append(slam.NODE_TYPE.POSE)

            edge_pose_to_pose = o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=idx_current_pose,
                target_node_id=idx_previous_pose,
                transformation=relative_transform,
                uncertain=True,
            )
            pose_graph.edges.append(edge_pose_to_pose)
            edge_types.append(slam.EDGE_TYPE.ODOMETRY)

        # register landmarks
        # cone_coordinates = np.asarray(current_frame.points)
        # for idx_cone, coord in enumerate(cone_coordinates):
        #     relative_cone_transform = np.eye(4)
        #     relative_cone_transform[:3, 3] = coord
        #     global_cone_transform = (
        #         pose_graph.nodes[idx_current_pose].pose @ relative_cone_transform
        #     )

        #     pose_graph.nodes.append(
        #         o3d.pipelines.registration.PoseGraphNode(global_cone_transform)
        #     )
        #     node_types.append(slam.NODE_TYPE.LANDMARK)

        #     edge_cone_to_pose = o3d.pipelines.registration.PoseGraphEdge(
        #         source_node_id=idx_current_pose + idx_cone + 1,
        #         target_node_id=idx_current_pose,
        #         transformation=relative_cone_transform,
        #         uncertain=True,
        #     )
        #     pose_graph.edges.append(edge_cone_to_pose)
        #     edge_types.append(slam.EDGE_TYPE.OBSERVATION)

        # register loop closure
        if idx == len(lidar_dataset) - 1:

            icp_result = o3d.pipelines.registration.registration_icp(
                current_frame,
                lidar_dataset[0],
                1.5,
                relative_transform,
            )
            relative_transform = icp_result.transformation

            edge_pose_to_pose = o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=0,
                target_node_id=idx_current_pose,
                transformation=relative_transform,
                uncertain=False,
            )
            pose_graph.edges.append(edge_pose_to_pose)
            edge_types.append(slam.EDGE_TYPE.LOOP_CLOSURE)

        ### (VISUAL) ### dynamic
        if DYNAMIC:
            pose_graph_dyn[idx].transform(pose_graph.nodes[idx_current_pose].pose)
            pose_graph_dyn[idx].paint_uniform_color([0, 0, 0])
            vis.add_geometry(pose_graph_dyn[idx])

            if SHOW_POSES:
                pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
                pose.transform(pose_graph.nodes[idx_current_pose].pose)
                poses = poses + pose

                vis.add_geometry(poses)

            if SHOW_EDGES:
                point = np.asarray(pose_graph.nodes[idx_current_pose].pose[:3, 3])
                edges.points.append(point)

                path.points.append(point)
                path.colors.append([1, 0.65, 0])
                vis.add_geometry(path)

                if idx > 0:
                    edges.lines.append([idx - 1, idx])
                    vis.add_geometry(edges)

                if idx == len(lidar_dataset) - 1:
                    edges.lines.append([idx, 0])
                    vis.add_geometry(edges)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)

    ### (VISUAL) ### dynamic
    if DYNAMIC:
        vis.destroy_window()

    ### (VISUAL) ### static
    if VISUALIZE or DYNAMIC:
        utils.draw(
            slam.generate_static_visualization(
                lidar_dataset,
                pose_graph,
                node_types,
            )
        )

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=1.5,
        edge_prune_threshold=0.25,
        reference_node=0,
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    ### (VISUAL) ### static
    if VISUALIZE or DYNAMIC:
        utils.draw(
            slam.generate_static_visualization(
                lidar_dataset,
                pose_graph,
                node_types,
            )
        )
