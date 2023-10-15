import numpy as np
import open3d as o3d
import argparse
import glob
import copy
from sklearn.neighbors import NearestNeighbors
import random

scale = 1
trans_list = []
image_total_number = 0
# max_correspondence_distance_coarse = voxel_size * 15
# max_correspondence_distance_fine = voxel_size * 1.5

def depth_image_to_point_cloud(rgb, depth):
    
    height, width = 512, 512
    cx = width/2
    cy = height/2
    fy = cy
    fx = cx

    cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, scale, depth_trunc=np.inf, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, cameraIntrinsics)
    print("pcd: \n", pcd)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 30
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 70
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=10)
    )
    print("voxel: \n", pcd_down)
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 15
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                1),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    threshold = voxel_size * 5
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):

    max_iterations = 300
    threshold = voxel_size * 5
    controlPoints = 100
    prev_error = 0

    source_down = source_down.transform(trans_init)
    source_points = np.asarray(source_down.points)
    A = source_points[:, :3]
    target_points = np.asarray(target_down.points)
    B = target_points[:, :3]

    if (A.shape[0] != B.shape[0]):
        length = min(A.shape[0], B.shape[0])
        length = min(length, controlPoints)
        sampleA = random.sample(range(A.shape[0]), length)
        sampleB = random.sample(range(B.shape[0]), length)
        P = np.array([A[i] for i in sampleA])
        Q = np.array([B[i] for i in sampleB])
    else:
        length = A.shape[0]
        if (length > controlPoints):
            sampleA = random.sample(range(A.shape[0]), length)
            sampleB = random.sample(range(B.shape[0]), length)
            P = np.array([A[i] for i in sampleA])
            Q = np.array([B[i] for i in sampleB])
        else:
            P = A
            Q = B

    m = P.shape[1]
    src = np.ones((m+1, P.shape[0]))
    dst = np.ones((m+1, Q.shape[0]))
    src[:m, :] = np.copy(P.T)
    dst[:m, :] = np.copy(Q.T)

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        src = np.dot(T, src)

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < threshold:
            break
        prev_error = mean_error

    T, _, _ = best_fit_transform(P, src[:m, :].T)
    # T = np.linalg.inv(T)
    return T


def reconstruct(args):
    if (args.version == 'open3d'):
        voxel_size = 0.003

    elif (args.version == 'my_icp'):
        voxel_size = 0.0025
    print("image number: ", image_total_number)
    batch = 10
    run_time = int(image_total_number/10)
    pcd_list = []
    result_pcd = o3d.geometry.PointCloud()
# ----------------------------------------------------------------------
    for i in range(1, run_time+1):
        print("Do batch number :", i)
        print("load image :", batch*(i-1)+1)
        color_source = o3d.io.read_image("%s/rgb/%s.png" % (args.data_root, batch*(i-1)+1))
        depth_source = o3d.io.read_image("%s/depth/%s.png" % (args.data_root, batch*(i-1)+1))
        source = depth_image_to_point_cloud(color_source, depth_source)
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        batch_pcd = o3d.geometry.PointCloud()
        batch_pcd += source_down
        for batch_number in range((batch*(i-1)+2), (batch*i+2)):
            print("load image :", batch_number)
            # TARGET frame:
            color_target = o3d.io.read_image(
                "%s/rgb/%s.png" % (args.data_root, batch_number))
            depth_target = o3d.io.read_image(
                "%s/depth/%s.png" % (args.data_root, batch_number))
            target = depth_image_to_point_cloud(color_target, depth_target)
            target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
            # ----------------------------------------------------------------------
            result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh, voxel_size)

            if (args.version == 'open3d'):
                trans = local_icp_algorithm(
                    source_down, target_down, result_ransac.transformation, voxel_size)
                transformation_matrix = trans.transformation
            elif (args.version == 'my_icp'):
                trans = my_local_icp_algorithm(
                    source_down, target_down, result_ransac.transformation, voxel_size)
                transformation_matrix = np.asarray(trans)
            # ----------------------------------------------------------------------
                #trans_list.append(transformation_matrix)
            # transformation_matrix = result_ransac.transformation
            # if batch_number % 5 == 0:
            #     draw_registration_result(
            #         source_down, target_down, transformation_matrix)
            batch_pcd = batch_pcd.transform(transformation_matrix)
            batch_pcd += target_down
            batch_pcd, batch_pcd_fpfh = preprocess_point_cloud(batch_pcd, voxel_size)
            source_down, source_fpfh = batch_pcd, batch_pcd_fpfh
        pcd_list.append(batch_pcd)
# -------------------------------------------------------------------------------        
    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 1500000
    max_correspondence_distance_fine = voxel_size * 150000
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcd_list,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
        
    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold= 1000000,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    for point_id in range(len(pcd_list)):
        print(pose_graph.nodes[point_id].pose)
        pcd_list[point_id].transform(pose_graph.nodes[point_id].pose)
        result_pcd += pcd_list[point_id]


    result_pcd = result_pcd.voxel_down_sample(0.001)
    result_pcd = limit_pcd_height(result_pcd)

    return result_pcd,  # pred_cam_pos

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def limit_pcd_height(pcd):
    # Convert PointCloud to numpy array for easy manipulation
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    max_height = 2.5
    min_height = 0
    # Filter points based on height
    mask = np.logical_and(points[:, 1] >= min_height,
                          points[:, 1] <= max_height)
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Create a new PointCloud with filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd


def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def best_fit_transform(A, B):

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str,
                        default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str,
                        default='data_collection/first_floor/')
    args = parser.parse_args()

    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # import images
    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"

    file_list = glob.glob("%s./rgb/*" % (args.data_root))
    image_total_number = len(file_list)

    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    # result_pcd, pred_cam_pos = reconstruct()
    result_pcd = reconstruct(args)

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    print("Mean L2 distance: ", )

    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    o3d.visualization.draw_geometries(result_pcd)
