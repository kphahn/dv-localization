import numpy as np
import open3d as o3d


def preprocess_frame(frame):

    min_bound = np.array([-7, 0, -0.5])
    max_bound = np.array([7, 20, 1])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    frame = frame.crop(bounding_box)
    frame = frame.voxel_down_sample(0.05)

    return frame


def remove_ground(frame, seed):

    # make function deterministic
    o3d.utility.random.seed(seed)

    _, inliers = frame.segment_plane(
        distance_threshold=0.05, ransac_n=5, num_iterations=1000
    )

    return frame.select_by_index(inliers, invert=True)


def extract_clusters(frame, min_detection_points):

    # cluster remaining pointcloud and get cone centers
    labels = np.array(
        frame.cluster_dbscan(
            eps=0.285 / 2,  # half a cone
            min_points=min_detection_points,
        )
    )
    max_label = labels.max()

    clusters = [
        frame.select_by_index(np.where(labels == i)[0]).paint_uniform_color(
            np.asarray([1, 0.3, 0])
        )
        for i in range(max_label + 1)
    ]

    return clusters


def filter_clusters(clusters):

    cones = []

    for cluster in clusters:
        bb = cluster.get_axis_aligned_bounding_box()
        if bb.get_extent()[2] > 0.1:
            cones.append(cluster)

    return cones


def get_cone_coordinates(cones):

    # derive cone positions from highest point of cluster and place on ground
    coordinates = []
    for cone in cones:
        points = np.asarray(cone.points)
        max_z_index = np.argmax(points[:, 2])
        highest_point = points[max_z_index]
        highest_point[2] = 0

        coordinates.append(highest_point)

    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coordinates))