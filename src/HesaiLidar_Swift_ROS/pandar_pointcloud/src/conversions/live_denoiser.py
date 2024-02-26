#!/usr/bin/env python3
import rospy
import pcl
import sensor_msgs.point_cloud2 as pc2
import numpy as np
np.float = np.float64         
from sensor_msgs.msg import PointCloud2
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2
from sklearn.neighbors import KDTree


def count_neighbors_kdtree(points, radius):
    neighbor_counts = []
    tree = KDTree(points, leaf_size=20)
    all_nn_indices = tree.query_radius(points, r=radius)
    # neighbor_counts = [len(nn_indices) for nn_indices in all_nn_indices]
    for i, nn_indices in enumerate(all_nn_indices):
        neighbor_counts.append(len(nn_indices))
    return neighbor_counts  

def count_neighbors_pcl(points, search_radius):
    # print('input cloud shape = ', cloud.shape)
    neighbor_count = []
    cloud = pcl.PointCloud(points)
    kdtree = cloud.make_kdtree()

    # Create a Euclidean Cluster Extraction object
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(search_radius)  # was 0.2 
    ec.set_MinClusterSize(100) # was 100 # was 50 
    ec.set_MaxClusterSize(5000) # was 5000 
    ec.set_SearchMethod(kdtree) # was kdtree

    # Call the extraction function to obtain cluster indices
    cluster_indices = ec.Extract()
    
    return cluster_indices

def count_neighbors_spatial_hash(points, search_radius): # search_radii
    spatial_hash = {}

    # points = np.round(points).astype(np.int32)
    # search_radii = np.round(search_radii).astype(np.int32)

    # for i, (point, search_radius) in enumerate(zip(points, search_radii)):
    for i, point in enumerate(points):
        if search_radius == 0: # DE VOOR LOOP SCHIET HIER VAAK IN, DUS FF GOED CHECKEN!
            # print(search_radius)
            print("Search radius = 0 skip")
            continue # Skip this point if search_radius is zero
    
        # Check for NaN values in the point
        if np.any(np.isnan(point)):
            print("NaN skip")
            continue # Skip this point if it contains NaN values

        cell = tuple(int(np.rint(coord / search_radius)) for coord in point)
        if cell not in spatial_hash:
            spatial_hash[cell] = []
        spatial_hash[cell].append(i)
    num_neighbors = [len(spatial_hash.get(tuple(int(coord / search_radius) for coord in point), [])) for point in points]
    num_neighbors = np.array(num_neighbors)
    return num_neighbors


def remove_zero_vecs(points):
    mask = (points['x'] == 0) & (points['y'] == 0) & (points['z'] == 0)
    non_zero_points = points[~mask]
    return non_zero_points 

def align_cloud(points):
    p_xyz = np.vstack([points['x'], points['y'], points['z']])
    p_itr = np.vstack([points['intensity'], points['timestamp'], points['ring']])
    angle_radians = np.radians(42) # set angle adjustment
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                [np.sin(angle_radians), np.cos(angle_radians), 0],
                                [0, 0, 1]])
    p_xyz = np.dot(p_xyz.T, rotation_matrix.T)
    transformed_cloud = []
    # print(p_xyz.shape, p_itr.shape)
    for i, (query_point, itr_point) in enumerate(zip(p_xyz, p_itr.T)):
        transformed_point = (query_point[0], query_point[1], query_point[2], itr_point[0], itr_point[1], itr_point[2])
        transformed_cloud.append(transformed_point)
         
    custom_dtype = np.dtype({'names':['x','y','z','intensity','timestamp','ring'], 'formats':['<f4','<f4','<f4','u1','<f8','<u2'], 'offsets':[0,4,8,16,24,32], 'itemsize':48})
    transformed_cloud = np.array(transformed_cloud, dtype=custom_dtype)
    return transformed_cloud


def dynamic_intensity_thresh(points, threshold_factor):
    max_z = np.max(points['z'])

    # Dit nog even SLIMMER doen maar werkt wel redelijk goed nu 
    mask = points['z'] <= (max_z - 0.7)
    points = points[mask]
    
    p_xyz = np.vstack([points['x'], points['y'], points['z']])
    p_itr = np.vstack([points['intensity'], points['timestamp'], points['ring']])

    # algining point cloud with ground plane using homogenous transform
    p_xyz = p_xyz.T

    # p_xyz = p_xyz.T
    p_itr = p_itr.T
    p_dist = np.linalg.norm(p_xyz, axis=1) 
    # Ensure that points and p_dist have the same length
    assert len(points) == len(p_dist), "Mismatched lengths of points and p_dist arrays."

    # Calculate intensity thresholds based on distance 
    intensity_thresholds = p_dist*threshold_factor
  
    # Initialize arrays for points below and above the intensity thresholds
    p_i_low = []
    p_o = []

    for index, (query_point, itr_point, i_value) in enumerate(zip(p_xyz, p_itr, points['intensity'])):
        intensity_threshold = intensity_thresholds[index]
        point_intensity = i_value

        if point_intensity < threshold_factor: #intensity_threshold: 
            low_point = (query_point[0], query_point[1], query_point[2], itr_point[0], itr_point[1], itr_point[2])
            p_i_low.append(low_point)

        else: 
            outlier_point = (query_point[0], query_point[1], query_point[2], itr_point[0], itr_point[1], itr_point[2])
            p_o.append(outlier_point)
         
    custom_dtype = np.dtype({'names':['x','y','z','intensity','timestamp','ring'], 'formats':['<f4','<f4','<f4','u1','<f8','<u2'], 'offsets':[0,4,8,16,24,32], 'itemsize':48})
    
    # Convert the lists to numpy arrays
    p_i_low = np.array(p_i_low, dtype=custom_dtype)
    p_o = np.array(p_o, dtype=custom_dtype)

    p_i_low = remove_zero_vecs(p_i_low)
    return p_i_low, p_o   

def denoiser(points): # AGDOR rebuild

    # --- set params --- 
    # intensity_thresh =  45 # np.median(points['intensity']) # was 35
    alpha = 0.5 # you may adjust this hyper parameter 
    angular_res = 0.6 # is between 0.4 and 0.8 depending on the layer
    min_neighbors = 5 # 15
    
    # --- stage 1 --- intensity filter (should be based labelled data of different dataset - IS NOT ATM!)
    # Check if intensity information is available
    if 'intensity' not in points.dtype.names:
        # No intensity information, handle accordingly
        rospy.loginfo(points.shape)
        rospy.logwarn("Intensity information not found in point cloud.")
        return points


    # --- stage 1 --- intensity filter (should be based labelled data of different dataset - IS NOT ATM!)
    q1 = np.percentile(points['intensity'], 25)
    q3 = np.percentile(points['intensity'], 75)
    # print("intensity thresh value: ", q3)

    mask = points['intensity'] < q3
    p_o = points[~mask] # bevast GEEN null values
    p_i_low = points[mask] # bevat veel null values 

    # p_i_low , p_o = dynamic_intensity_thresh(points, q3) # 35 DEZE GEBRUIKTE IK! 
    # p_i_low , p_o = dynamic_intensity_thresh(points, 20) # 35

    # return p_o


    # print("shape p_i_low = ", p_i_low.shape)
    # print("shape p_o = ", p_o.shape)
    # print("Percentage of p_i_low points = ", round(p_i_low.shape[0]/points.shape[0], 2))

    # --- stage 2 --- adaptive group density outlier removal
    p_xyz = np.vstack([p_i_low['x'], p_i_low['y'], p_i_low['z']])
    p_itr = np.vstack([p_i_low['intensity'], p_i_low['timestamp'], p_i_low['ring']])

    p_xyz = p_xyz.T
    p_itr = p_itr.T

    # print("Number of p_i_low zero vectors:", count_zero_vecs(p_i_low))
    p_dist = np.linalg.norm(p_xyz, axis=1) 

    SR = alpha*angular_res*p_dist # apply a dynamic search radius 

    # print("median SR = ", np.median(SR))
    # print("mean SR = ", np.mean(SR))
    # # print(np.unique(p_dist, return_counts=True))
    # print("max p_dist value = ", np.max(p_dist))

    p_core = []
    p_outlier = []


    # num_neighbors_array = count_neighbors_kdtree(p_xyz, search_radiusje) # was search_radiusje
    # num_neighbors_array = count_neighbors_spatial_hash(p_xyz, 0.01) # was np.median(SR)
    core_indices = count_neighbors_pcl(p_xyz, 0.10) # was 0.10 # was 0.05
    core_indices = np.hstack(core_indices)
    p_core = p_i_low[core_indices]

 
    if np.size(p_core)!=0:
        p_stacked = np.concatenate((p_o, p_core), axis=0)
        unique_points, indices = np.unique(p_stacked, axis=0, return_index=True)
        p_o = p_stacked[np.sort(indices)]
        return p_o
    else:
        print("WARNING: the SR did not return any points!")
        return p_o 
    
frames = 0
def callback(raw_cloud):
    global frames
    rospy.loginfo("Processing the point cloud")

    # Convert the raw PointCloud2 to a NumPy array
    cloud_array = pointcloud2_to_array(raw_cloud)
    cloud_array = remove_zero_vecs(cloud_array)
    # cloud_array = align_cloud(cloud_array)

    # Denoise the raw point cloud
    filtered_cloud_array = denoiser(cloud_array)
    
    # Convert denoised cloud for publishing
    filtered_cloud_msg = array_to_pointcloud2(filtered_cloud_array, stamp=raw_cloud.header.stamp, frame_id="PandarSwift")
    
    # Publish the denoised point cloud
    pub.publish(filtered_cloud_msg)
    frames += 1
    print(frames)


if __name__ == '__main__':
    rospy.init_node('denoising_node', anonymous=True)
    sub = rospy.Subscriber('hesai/pandar_points', PointCloud2, callback) # live_cloud # hesai/pandar_points reading the raw point cloud topic for data to be proccessed, was: hesai/pandar_points
    pub = rospy.Publisher('denoised_cloud', PointCloud2, queue_size=10) # this is the topic to which the processed point gets publsihed
    rate = rospy.Rate(10)  # Adjust the rate as needed

    rospy.spin()