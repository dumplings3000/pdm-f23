import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    """
    Preprocess the point cloud by downsampling it using voxel grid filtering.

    Parameters:
    - pcd: input point cloud
    - voxel_size: voxel size for downsampling

    Returns:
    - downsampled point cloud
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down

def registration_icp(source, target, voxel_size=0.05, distance_threshold=0.02):
    """
    Perform ICP registration between source and target point clouds.

    Parameters:
    - source: source point cloud
    - target: target point cloud
    - voxel_size: voxel size for downsampling (optional, default=0.05)
    - distance_threshold: distance threshold for ICP (optional, default=0.02)

    Returns:
    - registration result
    """
    # Downsample the point clouds
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)

    # Estimate normals for both point clouds
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Perform global registration (e.g., RANSAC)
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_down.estimate_normals(), target_down.estimate_normals(),
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # Maximum number of correspondences to consider for each point
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    # Perform ICP registration using the result from global registration
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    return result_icp

# Load your point clouds (replace these with your actual point cloud files)
source = o3d.io.read_point_cloud("path/to/source.pcd")
target = o3d.io.read_point_cloud("path/to/target.pcd")

# Perform ICP and global registration
result = registration_icp(source, target)

# Visualize the result
o3d.visualization.draw_geometries([source, target, result.transient_point_cloud])
