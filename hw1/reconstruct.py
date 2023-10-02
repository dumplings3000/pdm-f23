import numpy as np
import open3d as o3d
import argparse
import cv2
import glob
import os


pcd_list = []


def depth_image_to_point_cloud(rgb, depth):
    # Get point cloud from rgb and depth image
    pcd = []
    colors = []
    scale = 1000

    # camera internal parameter
    # fov = 90 degree and the camera resolution is 512 * 512,depth_scale = 1000
    height, width = depth.shape
    cx = width/2
    cy = height/2
    fy = cy
    fx = cx

    # pixels with depth to point cloud
    for i in range(height):
        for j in range(width):
            z = depth[i, j]
            y = (-i+cy) * z / fy
            x = (-j+cx) * z / fx
            color = rgb[i, j]/255
            pcd.append([x, y, z])
            colors.append([color[2], color[1], color[0]])

    # create point cloud object
    pcd_o3d = o3d.geometry.PointCloud() 
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

    return pcd_o3d

def preprocess_point_cloud(pcd, voxel_size = 0.009):
    
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                                                                pcd_down,
                                                                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
                                                                )
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size = 0.009):
    distance_threshold = voxel_size * 1.25
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                                                                                    source_down, target_down, source_fpfh, target_fpfh, True,                                                                                        distance_threshold,
                                                                                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
                                                                                    [
                                                                                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                                                                                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                                                                                    ], 
                                                                                    o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
                                                                                    )
    return result
 
def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    # Run the ICP registration
    result = icp.registration_result.transform
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Write your own ICP function
    raise NotImplementedError
    return result


def reconstruct(args):

    # if(args.version == 'open3d'):
    #     trans = local_icp_algorithm()
    # elif(args.version == 'my_icp'):
    #     trans = my_local_icp_algorithm()

    for i in range(len(rgb_file_list)):
        rgb = cv2.cvtColor(cv2.imread(rgb_file_list[i], 3), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_file_list[i], -1)
        pcd = depth_image_to_point_cloud(rgb, depth)
        pcd_list.append(pcd)

    for i in range (len(pcd_list)):
        source = pcd_list[i+1]
        target = pcd_list[i]
        source_down, source_fpfh = preprocess_point_cloud(source)
        target_down, target_fpfh = preprocess_point_cloud(target)
        global_registration = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh)
        result_pcd = global_registration

    return result_pcd, #pred_cam_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str,
                        default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str,
                        default='data_collection/first_floor/')
    args = parser.parse_args()
    #import images
    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"

    rgb_file_list = glob.glob(os.path.join(args.data_root,'rgb','*'))
    depth_file_list = glob.glob(os.path.join(args.data_root,'depth','*'))
    rgb_file_list = [os.path.join(args.data_root,'rgb/{}.png').format(i+1) for i in range(len(rgb_file_list))]
    depth_file_list = [os.path.join(args.data_root,'depth/{}.png').format(i+1) for i in range(len(depth_file_list))]

    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    #result_pcd, pred_cam_pos = reconstruct()
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
