import numpy as np
import open3d as o3d
import argparse
import cv2
import copy

scale = 1
voxel_size = 0.005
trans_list = []
pcd_list =[]

def depth_image_to_point_cloud(rgb, depth):
    # Get point cloud from rgb and depth image
    # camera internal parameter
    # fov = 90 degree and the camera resolution is 512 * 512,depth_scale = 1000
    height, width = 512, 512
    cx = width/2
    cy = height/2
    fy = cy
    fx = cx

    cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, scale,
        depth_trunc=np.inf,
        convert_rgb_to_intensity=False)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cameraIntrinsics)
    print("pcd: \n",pcd)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 30
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 70
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                                                                pcd_down,
                                                                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
                                                                )
    print("voxel: \n",pcd_down)
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
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99))
    return result
 
def local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    threshold = voxel_size * 4
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000))
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Write your own ICP function
    raise NotImplementedError
    return result

def reconstruct(args):
    image_total_number = 141
    batch = int(image_total_number/2)
#----------------------------------------------------------------------
    print("Do batch number :",1)
    print("load image :", 1)
    color_source = o3d.io.read_image("%s/rgb/%s.png"%(args.data_root, 1))
    depth_source = o3d.io.read_image("%s/depth/%s.png"%(args.data_root, 1))
    source = depth_image_to_point_cloud(color_source, depth_source)
    source_down, source_fpfh = preprocess_point_cloud(source,voxel_size)
    batch_pcd = o3d.geometry.PointCloud()
    batch_pcd += source
    for batch_number in range (2,batch+1):
        print("load image :",batch_number)
        # TARGET frame:
        color_target = o3d.io.read_image("%s/rgb/%s.png"%(args.data_root, batch_number))
        depth_target = o3d.io.read_image("%s/depth/%s.png"%(args.data_root, batch_number))
        target = depth_image_to_point_cloud(color_target, depth_target)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        #----------------------------------------------------------------------
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh, voxel_size)
            
        if(args.version == 'open3d'):
            trans = local_icp_algorithm(source_down, target_down, result_ransac.transformation, voxel_size)
        elif(args.version == 'my_icp'):
            trans = my_local_icp_algorithm(source_down, target_down, result_ransac.transformation, voxel_size)
        #----------------------------------------------------------------------
        trasformation_matrix = trans.transformation
        #print(trasformation_matrix)
        #trans_list.append(trasformation_matrix)
        #draw_registration_result(source_down,target_down,trasformation_matrix)
        batch_pcd = batch_pcd.transform(trasformation_matrix)
        batch_pcd += target
        source = target
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

    pcd_list.append(batch_pcd)
#--------------------------------------------------------------------------------
    print("Do batch number :", 2)
    print("load image :", 141)
    color_source = o3d.io.read_image("%s/rgb/%s.png"%(args.data_root, 141))
    depth_source = o3d.io.read_image("%s/depth/%s.png"%(args.data_root, 141))
    source = depth_image_to_point_cloud(color_source, depth_source)
    source_down, source_fpfh = preprocess_point_cloud(source,voxel_size)
    batch_pcd = o3d.geometry.PointCloud()
    batch_pcd += source
    
    for batch_number in range (1,batch+2):
        
        print("load image :", 141-batch_number)
        # TARGET frame:
        color_target = o3d.io.read_image("%s/rgb/%s.png"%(args.data_root, 141-batch_number))
        depth_target = o3d.io.read_image("%s/depth/%s.png"%(args.data_root, 141-batch_number))
        target = depth_image_to_point_cloud(color_target, depth_target)
        #target = limit_pcd_height(target)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        #----------------------------------------------------------------------
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh, voxel_size)
            
        if(args.version == 'open3d'):
            trans = local_icp_algorithm(source_down, target_down, result_ransac.transformation, voxel_size)
        elif(args.version == 'my_icp'):
            trans = my_local_icp_algorithm(source_down, target_down, result_ransac.transformation, voxel_size)
        #----------------------------------------------------------------------
        trasformation_matrix = np.linalg.inv(trans.transformation)
        #print(trasformation_matrix)
        #trans_list.append(trasformation_matrix)
        #draw_registration_result(source_down,target_down,trasformation_matrix)
        batch_pcd = batch_pcd.transform(trasformation_matrix)
        batch_pcd += target
        source = target
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

    pcd_list.append(batch_pcd)

    result_pcd = o3d.geometry.PointCloud()
    result_pcd += pcd_list[0]

    for i in range(len(pcd_list)-1):
        print("merge %s batch and %s batch :",i+1,i+2)
        source = pcd_list[i]
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down = pcd_list[i+1]
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh, voxel_size)
            
        if(args.version == 'open3d'):
                trans = local_icp_algorithm(source_down, target_down, result_ransac.transformation, voxel_size)
        elif(args.version == 'my_icp'):
                trans = my_local_icp_algorithm(source_down, target_down, result_ransac.transformation, voxel_size)
            #----------------------------------------------------------------------
        trasformation_matrix = trans.transformation
        draw_registration_result(source_down,target_down,trasformation_matrix)
        result_pcd = result_pcd.transform(trasformation_matrix)
        result_pcd += target
        source = target
        source_down, source_fpfh = target_down, target_fpfh
       

    result_pcd = limit_pcd_height(result_pcd)
    print(result_pcd)
       
    return result_pcd, #pred_cam_pos


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
    mask = np.logical_and(points[:, 1] >= min_height, points[:, 1] <= max_height)
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Create a new PointCloud with filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str,
                        default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str,
                        default='data_collection/first_floor/')
    args = parser.parse_args()

    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    #import images
    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"


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
