import numpy as np
import open3d as o3d
import argparse
import cv2
import glob
import os

pcd = []
colors = []


def depth_image_to_point_cloud(rgb, depth):
    # Get point cloud from rgb and depth image

    # get rgb and depth images
    scale = 1000
    
    # camera internal parameter
    # fov = 90 degree and the camera resolution is 512 * 512,depth_scale = 1000
    height, width = .shape
    cx = width/2
    cy = height/2
    fy = cy
    fx = cx

    # pixels with depth to point cloud
    for i in range(height):
        for j in range(width):
            z = depth_image[i, j]
            y = (-i+cy) * z / fy
            x = (-j+cx) * z / fx
            color = rgb_image[i, j]/255
            pcd.append([x, y, z])
            colors.append([color[2], color[1], color[0]])

    # create point cloud object
    pcd_o3d = o3d.geometry.PointCloud() 
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    print(pcd_o3d)

    return pcd_o3d

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return pcd_down

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    raise NotImplementedError
    return result

def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    # Run the ICP registration
    result = icp.registration_result.transform
    return result


# def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
#     # TODO: Write your own ICP function
#     raise NotImplementedError
#     return result


# def reconstruct(args):

#     source_pcd = depth_image_to_point_cloud(args.data_root + "/rgb/1.png", args.data_root + "/depth/1.png")
#     target_pcd = depth_image_to_point_cloud(args.data_root + "/rgb/2.png", args.data_root + "/depth/2.png")
#     source_down = preprocess_point_cloud(source_pcd, 0.0005)
#     target_down = preprocess_point_cloud(target_pcd, 0.0005)
#     trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
#                              [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#     threshold = 0.02
#     result_pcd = local_icp_algorithm(source_down, target_down, trans_init, threshold)

#     return result_pcd, pred_cam_pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str,
                        default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str,
                        default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"

    rgb_file_list = glob.glob(os.path.join(args.data_root,'rgb','*'))
    depth_file_list = glob.glob(os.path.join(args.data_root,'depth','*'))
    rgb_file_list = ["./rgb/{}.png".format(i+1) for i in range(len(rgb_file_list))]
    depth_file_list = ["./depth/{}.png".format(i+1) for i in range(len(depth_file_list))]


    # TODO: Output result point cloud and estimated camera pose
    pcd_o3d = depth_image_to_point_cloud(args.data_root + "/rgb/1.png", args.data_root + "/depth/1.png")
    pcd_down = preprocess_point_cloud(pcd_o3d, 0.0009)
    #icp = reconstruct(args)
    # Visualize:
    o3d.visualization.draw_geometries([pcd_o3d])
    