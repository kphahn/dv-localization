import open3d as o3d
import numpy as np

import utils


def associate_landmarks(
    new_landmarks,
    known_landmarks,
    initial_transform,
):

    icp_result = o3d.pipelines.registration.registration_icp(
        new_landmarks,
        known_landmarks,
        0.285,
        initial_transform,
    )

    return icp_result.correspondence_set


def estimate_odometry(
    source_frame,
    target_frame,
    initial_transform,
):

    if source_frame == None or target_frame == None:
        return np.eye(4)

    icp_result = o3d.pipelines.registration.registration_icp(
        source_frame,
        target_frame,
        0.5,
        initial_transform,
    )

    return icp_result.transformation


def optimize_pose_graph(pose_graph):

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

    current_map = pose_graph.get_map()
    labels = np.array(
        current_map.cluster_dbscan(
            eps=0.5,
            min_points=1,
        )
    )
    max_label = labels.max()

    clusters = [
        current_map.select_by_index(np.where(labels == i)[0])
        for i in range(max_label + 1)
    ]

    # new_pose_graph = PoseGraph(np.eye(4))
    new_pose_graph = PoseGraph(pose_graph.current_pose)
    new_pose_graph.previous_frame = pose_graph.previous_frame

    centers = o3d.utility.Vector3dVector([cluster.get_center() for cluster in clusters])
    for center in centers:
        center[2] = 0
        position = np.eye(4)
        position[:3, 3] = center

        new_pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(position))
        new_pose_graph.idxs_landmarks.append(len(new_pose_graph.nodes) - 1)

        new_pose_graph.lines.points.append(np.asarray(position[:3, 3]))

        new_pose_graph.map.points.append(position[:3, 3])
        # new_pose_graph.map.colors.append(np.asarray([0, 0, 0]))

    color = utils.generate_colors(1)[0]
    previous_path = pose_graph.get_path().paint_uniform_color(np.asarray(color))

    return new_pose_graph, previous_path


class PoseGraph(o3d.pipelines.registration.PoseGraph):

    def __init__(
        self,
        coordinate_offset=np.eye(4),
    ):

        super().__init__()

        self.previous_frame = None

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

    def add_edge(
        self,
        idx_source_node,
        idx_target_node,
        transformation,
        uncertain,
        information=np.eye(6),
    ):

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
