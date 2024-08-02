import open3d as o3d
import numpy as np
import time


def associate_landmarks(new_landmarks, known_landmarks, initial_transform):

    icp_result = o3d.pipelines.registration.registration_icp(
        new_landmarks,
        known_landmarks,
        1,
        initial_transform,
    )

    # end_time = time.time()
    # print(f"estimate_odometry runtime: {end_time - start_time:.4f} seconds")

    return icp_result.correspondence_set


def estimate_odometry(
    source_frame,
    target_frame,
    max_correspondenc_dist,
    initial_transform,
):

    # start_time = time.time()

    icp_result = o3d.pipelines.registration.registration_icp(
        source_frame,
        target_frame,
        max_correspondenc_dist,
        initial_transform,
    )

    # end_time = time.time()
    # print(f"estimate_odometry runtime: {end_time - start_time:.4f} seconds")

    return icp_result.transformation


class PoseGraph(o3d.pipelines.registration.PoseGraph):

    def __init__(
        self,
        coordinate_offset=np.eye(4),
    ):

        super().__init__()

        self.nodes.append(o3d.pipelines.registration.PoseGraphNode(coordinate_offset))
        self.idxs_poses = [0]
        self.path = o3d.geometry.PointCloud()
        self.path.points.append(coordinate_offset[:3, 3])
        self.path.colors.append(np.asarray([1, 0, 0]))

        self.idxs_landmarks = []
        self.map = o3d.geometry.PointCloud()

        self.current_pose = coordinate_offset

        # for visualization
        self.lines = o3d.geometry.LineSet()
        self.lines.points.append(np.asarray(coordinate_offset[:3, 3]))

    def add_pose(
        self,
        relative_transform,
        landmark=False,
        color=np.asarray([0, 0, 0]),
    ):

        # start_time = time.time()

        global_transform = self.nodes[self.idxs_poses[-1]].pose @ relative_transform
        self.nodes.append(o3d.pipelines.registration.PoseGraphNode(global_transform))

        self.lines.points.append(np.asarray(global_transform[:3, 3]))

        if not landmark:
            self.idxs_poses.append(len(self.nodes) - 1)

            self.path.points.append(global_transform[:3, 3])
            self.path.colors.append(np.asarray([1, 0, 0]))

            self.current_pose = self.nodes[self.idxs_poses[-1]].pose

            self.add_edge(
                idx_source_node=self.idxs_poses[-2],
                idx_target_node=self.idxs_poses[-1],
                transformation=relative_transform,
                uncertain=True,
            )

        else:
            self.idxs_landmarks.append(len(self.nodes) - 1)

            self.map.points.append(global_transform[:3, 3])
            self.map.colors.append(color)

            self.add_edge(
                idx_source_node=self.idxs_poses[-1],
                idx_target_node=self.idxs_landmarks[-1],
                transformation=relative_transform,
                uncertain=True,
            )

        # end_time = time.time()
        # print(f"add_pose runtime: {end_time - start_time:.4f} seconds")

    def add_edge(
        self,
        idx_source_node,
        idx_target_node,
        transformation,
        uncertain,
        color=np.asarray([0, 0, 0]),
    ):

        # start_time = time.time()

        # edges need to be created "backwards", due to how open3d library is implemented
        self.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=idx_target_node,
                target_node_id=idx_source_node,
                transformation=transformation,
                uncertain=uncertain,
            )
        )

        self.lines.lines.append([idx_source_node, idx_target_node])
        self.lines.colors.append(color)

        # end_time = time.time()
        # print(f"add_edge runtime: {end_time - start_time:.4f} seconds")

    def add_loop_closure(
        self,
        idx_source_node,
        idx_target_node,
    ):

        T_source = self.nodes[idx_source_node].pose
        T_target = self.nodes[idx_target_node].pose

        T_relative = T_target @ np.linalg.inv(T_source)

        self.add_edge(
            idx_source_node,
            idx_target_node,
            T_relative,
            False,
            np.asarray([1, 0, 0]),
        )

    def opitimize(self):

        print(self.nodes[self.idxs_poses[-1]].pose)

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=1.5,
            edge_prune_threshold=0.25,
            reference_node=0,
        )

        o3d.pipelines.registration.global_optimization(
            self,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        print(self.nodes[self.idxs_poses[-1]].pose)

        # self._update_map()
        self._update_path()

    def get_map(self):

        map_points = [self.nodes[idx].pose[:3, 3] for idx in self.idxs_landmarks]
        map = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(map_points))

        return map

    def get_path(self):

        path_points = [
            np.asarray(self.nodes[idx].pose[:3, 3]) for idx in self.idxs_poses
        ]
        path = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(path_points))
        path.paint_uniform_color(np.asarray([1, 0, 0]))

        return path

    # frames = copy.deepcopy(frames)
    # poses = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # edges = o3d.geometry.LineSet()
    # path = o3d.geometry.PointCloud()

    # for idx, frame in enumerate(frames):

    #     idx_current_pose = pose_graph.idxs_poses[idx]
    #     pose_transform = pose_graph.nodes[idx_current_pose].pose

    #     frame.transform(pose_transform)

    #     if visualize_poses:
    #         pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #         pose.transform(pose_transform)
    #         poses = poses + pose

    #     if visualize_edges:
    #         point = np.asarray(pose_transform[:3, 3])
    #         edges.points.append(point)

    #         path.points.append(point)
    #         path.colors.append([1, 0.65, 0])

    #         if idx > 0:
    #             edges.lines.append([idx - 1, idx])
    #             edges.colors.append([0, 0, 0])

    #         if idx == len(frames) - 1:
    #             edges.lines.append([idx, 0])
    #             edges.colors.append([1, 0, 0])

    # pointcloud = []

    # if len(show_frames):
    #     for n in show_frames:
    #         pointcloud.append(frames[n])
    # else:
    #     pointcloud = frames

    # if visualize_poses:
    #     pointcloud.append(poses)

    # if visualize_edges:
    #     pointcloud.append(edges)
    #     pointcloud.append(path)

    # return pointcloud

    def visualize(
        self,
        show_map=True,
        show_path=True,
        show_edges=False,
    ):

        visual_set = []

        if show_map:
            map_points = [
                np.asarray(self.nodes[idx].pose[:3, 3]) for idx in self.idxs_landmarks
            ]
            map = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(map_points))

            visual_set.append(map)

        if show_path:
            visual_set.append(self.get_path())

        if show_edges and len(self.lines.lines):
            visual_set.append(self.lines)

        o3d.visualization.draw(
            visual_set,
            show_skybox=False,
            point_size=5,
        )
