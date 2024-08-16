import open3d as o3d
import numpy as np
import time

import utils
import copy


def associate_landmarks(new_landmarks, known_landmarks, initial_transform, idx):

    icp_result = o3d.pipelines.registration.registration_icp(
        new_landmarks,
        known_landmarks,
        0.5,
        initial_transform,
    )

    # if idx == 443:
    #     utils.draw(
    #         [
    #             copy.deepcopy(new_landmarks)
    #             .paint_uniform_color(np.asarray([1, 0, 0]))
    #             .transform(icp_result.transformation),
    #             known_landmarks,
    #         ]
    #     )

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
                uncertain=False,
            )

        else:
            self.idxs_landmarks.append(len(self.nodes) - 1)

            self.map.points.append(global_transform[:3, 3])
            self.map.colors.append(color)

            # ignore rotational information for landmark edges
            information = np.eye(6)
            information[:3, :3] = 0

            self.add_edge(
                idx_source_node=self.idxs_poses[-1],
                idx_target_node=self.idxs_landmarks[-1],
                transformation=relative_transform,
                uncertain=False,
                information=information,
            )

        # end_time = time.time()
        # print(f"add_pose runtime: {end_time - start_time:.4f} seconds")

    def add_edge(
        self,
        idx_source_node,
        idx_target_node,
        transformation,
        uncertain,
        information=np.eye(6),
        color=np.asarray([0, 0, 0]),
    ):

        # start_time = time.time()

        # edges need to be created "backwards", due to how open3d library is implemented
        self.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=idx_target_node,
                target_node_id=idx_source_node,
                transformation=transformation,
                information=information,
                uncertain=uncertain,
            )
        )

        self.lines.lines.append([idx_source_node, idx_target_node])
        self.lines.colors.append(color)

        # end_time = time.time()
        # print(f"add_edge runtime: {end_time - start_time:.4f} seconds")

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

    def get_map(self):

        map_points = [
            np.asarray(self.nodes[idx].pose[:3, 3]) for idx in self.idxs_landmarks
        ]
        self.map.points = o3d.utility.Vector3dVector(map_points)

        return self.map

    def get_path(self):

        path_points = [
            np.asarray(self.nodes[idx].pose[:3, 3]) for idx in self.idxs_poses
        ]

        self.path.points = o3d.utility.Vector3dVector(path_points)

        return self.path

    def get_lines(self):

        points = [
            np.asarray(self.nodes[idx].pose[:3, 3]) for idx in range(len(self.nodes))
        ]

        self.lines.points = o3d.utility.Vector3dVector(points)

        return self.lines

    def visualize(
        self,
        show_map=True,
        show_path=True,
        show_edges=False,
    ):

        visual_set = []

        if show_map:
            visual_set.append(self.get_map())

        if show_path:
            visual_set.append(self.get_path())

        if show_edges and len(self.edges):
            visual_set.append(self.get_lines())

        o3d.visualization.draw(
            visual_set,
            show_skybox=False,
            point_size=5,
        )
