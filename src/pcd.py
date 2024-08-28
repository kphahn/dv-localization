import open3d as o3d
import numpy as np


def crop_frame(frame, min_bound, max_bound):

    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return frame.crop(bounding_box)


def remove_ground(frame):

    _, inliers = frame.segment_plane(
        distance_threshold=0.01, ransac_n=5, num_iterations=1000
    )

    return frame.select_by_index(inliers, invert=True)


def detect_cones(frame, min_detection_points):

    # cluster remaining pointcloud and get cone centers
    labels = np.array(
        frame.cluster_dbscan(
            eps=0.1,
            min_points=min_detection_points,
        )
    )
    max_label = labels.max()

    clusters = [
        frame.select_by_index(np.where(labels == i)[0]) for i in range(max_label + 1)
    ]

    return clusters


def get_cone_coordinates(frame, min_detection_points, true_coordinates):

    # crop frame
    # min_bound = np.array([0, -5, -2])
    # max_bound = np.array([20, 5, 5])
    # frame = crop_frame(frame, min_bound, max_bound)

    # remove ground
    frame = remove_ground(frame)

    # detect cones
    cones = detect_cones(frame, min_detection_points)

    # derive cone positions from center and palce on ground
    coordinates = o3d.utility.Vector3dVector([cone.get_center() for cone in cones])
    for coord in coordinates:
        coord[2] = 0
    coordinates = o3d.geometry.PointCloud(coordinates)

    coordinates.colors = o3d.utility.Vector3dVector(
        np.zeros((len(coordinates.points), 3))
    )

    ### WORKAROUND FOR FAKESCENES DATASET - SHOULD BE REPLACED WITH PROPER CONE DETECTOR
    # Identify big cones in frame, if present
    tree = o3d.geometry.KDTreeFlann(coordinates)
    common_indices = []
    for point in true_coordinates["big_cones"].points:
        [_, idx, _] = tree.search_knn_vector_3d(point, 1)
        if np.linalg.norm(coordinates.points[idx[0]] - point) < 0.1:
            common_indices.append(idx[0])

    for idx in common_indices:
        coordinates.colors[idx] = np.asarray([1, 0.3, 0])
    ###

    return coordinates
