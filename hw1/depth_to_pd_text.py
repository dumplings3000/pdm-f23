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
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    # Visualize:
    o3d.visualization.draw_geometries([pcd_o3d])
    return pcd,colors


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
    depth_image_to_point_cloud(args.data_root + "/rgb/1.png",args.data_root + "/depth/1.png")
    

    #o3d.visualization.draw_geometries()
