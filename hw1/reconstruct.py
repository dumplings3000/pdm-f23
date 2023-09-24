import numpy as np
import open3d as o3d
import argparse
import cv2

pcd =[]
colors = []

def depth_image_to_point_cloud(rgb, depth):
     # Get point cloud from rgb and depth image 

    #get rgb and depth images
    scale = 1000
    rgb_image = cv2.imread(rgb)
    depth_image = cv2.imread(depth,cv2.IMREAD_UNCHANGED).astype(float) / scale


    # camera internal parameter
    # fov = 90 degree and the camera resolution is 512 * 512,depth_scale = 1000
    height, width = depth_image.shape
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
            color = rgb_image[i,j]/255
            pcd.append([x,y,z])
            colors.append([color[2],color[1],color[0]])
    return pcd,colors


def preprocess_point_cloud(pcd, voxel_size):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    raise NotImplementedError
    return pcd_down


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    raise NotImplementedError
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement
    raise NotImplementedError
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Write your own ICP function
    raise NotImplementedError
    return result


def reconstruct(args):
    # TODO: Return results
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    raise NotImplementedError
    return result_pcd, pred_cam_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    
    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    result_pcd, pred_cam_pos = reconstruct()

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
    o3d.visualization.draw_geometries()
