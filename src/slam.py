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
        0.4,
        initial_transform,
    )

    return icp_result.correspondence_set, icp_result.transformation


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
        1,
        initial_transform,
    )

    return icp_result.transformation


class PoseGraph(o3d.pipelines.registration.PoseGraph):

    def __init__(
        self,
        coordinate_offset=np.eye(4),
    ):

        super().__init__()

        self.nodes.append(o3d.pipelines.registration.PoseGraphNode(coordinate_offset))
        self.idxs_poses = [0]
        self.idxs_landmarks = []

        self.previous_frame = None
        self.current_pose = coordinate_offset

        # for visualization
        self.lines = o3d.geometry.LineSet()
        self.lines.points.append(np.asarray(coordinate_offset[:3, 3]))

    def add_pose(
        self,
        relative_transform,
        landmark=False,
    ):

        global_transform = self.nodes[self.idxs_poses[-1]].pose @ relative_transform
        self.nodes.append(o3d.pipelines.registration.PoseGraphNode(global_transform))

        self.lines.points.append(np.asarray(global_transform[:3, 3]))

        information = np.eye(6)

        if not landmark:
            self.idxs_poses.append(len(self.nodes) - 1)

            self.current_pose = self.nodes[self.idxs_poses[-1]].pose

            idx_source_node = self.idxs_poses[-2]
            idx_target_node = self.idxs_poses[-1]

        else:
            self.idxs_landmarks.append(len(self.nodes) - 1)

            # ignore rotational information for landmark edges
            information[:3, :3] = 0

            idx_source_node = self.idxs_poses[-1]
            idx_target_node = self.idxs_landmarks[-1]

        self.add_edge(
            idx_source_node=idx_source_node,
            idx_target_node=idx_target_node,
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

    def get_map(self):

        map_points = [
            np.asarray(self.nodes[idx].pose[:3, 3]) for idx in self.idxs_landmarks
        ]

        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(map_points))

    def get_path(self):

        path_points = [
            np.asarray(self.nodes[idx].pose[:3, 3]) for idx in self.idxs_poses
        ]

        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(path_points))

    def get_lines(self):

        points = [
            np.asarray(self.nodes[idx].pose[:3, 3]) for idx in range(len(self.nodes))
        ]

        self.lines.points = o3d.utility.Vector3dVector(points)
        self.lines.paint_uniform_color(utils.orange)
        self.lines.colors[-1] = utils.black

        return self.lines
