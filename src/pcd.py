import open3d as o3d
import numpy as np


def remove_ground(frame):

    _, inliers = frame.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=100
    )

    return frame.select_by_index(inliers, invert=True)


def detect_cones(frame, min_detection_points):

    frame = remove_ground(frame)

    # cluster remaining pointcloud and get cone centers
    labels = np.array(
        frame.cluster_dbscan(
            eps=0.1,
            min_points=min_detection_points,
        )
    )
    max_label = labels.max()

    cones = [
        frame.select_by_index(np.where(labels == i)[0]) for i in range(max_label + 1)
    ]

    return cones


def get_cone_coordinates(frame, min_detection_points, big_cones):

    # detect cones
    cones = detect_cones(frame, min_detection_points)

    # derive cone positions from center and set z = 0
    coordinates = o3d.utility.Vector3dVector([cone.get_center() for cone in cones])
    for coord in coordinates:
        coord[2] = 0
    coordinates = o3d.geometry.PointCloud(coordinates)

    ### WORKAROUND FOR FAKESCENES DATASET - SHOULD BE REPLACED WITH PROPER CONE DETECTOR
    # Identify big cones in frame, if present
    tree = o3d.geometry.KDTreeFlann(coordinates)
    common_indices = []
    for point in big_cones.points:
        [_, idx, _] = tree.search_knn_vector_3d(point, 1)
        if np.linalg.norm(coordinates.points[idx[0]] - point) < 0.1:
            common_indices.append(idx[0])
    ###

    # Highlight big cones with orange color
    coordinates.colors = o3d.utility.Vector3dVector(
        np.zeros((len(coordinates.points), 3))
    )
    for idx in common_indices:
        coordinates.colors[idx] = np.asarray([1, 0.3, 0])

    return coordinates
