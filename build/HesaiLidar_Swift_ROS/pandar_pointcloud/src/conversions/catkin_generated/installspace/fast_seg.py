#!/usr/bin/env python3
import rospy
import pcl
import sensor_msgs.point_cloud2 as pc2
import numpy as np
np.float = np.float64         
from sensor_msgs.msg import PointCloud2
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2


custom_dtype = np.dtype({'names':['x','y','z','intensity','timestamp','ring'], 'formats':['<f4','<f4','<f4','u1','<f8','<u2'], 'offsets':[0,4,8,16,24,32], 'itemsize':48})

def remove_zero_vecs(points):
    mask = (points['x'] == 0) & (points['y'] == 0) & (points['z'] == 0)
    non_zero_points = points[~mask]
    return non_zero_points 


def cluster_planes(cloud): 
    # print('input cloud shape = ', cloud.shape)
    cloud = pcl.PointCloud(cloud)
    kdtree = cloud.make_kdtree()

    # Create a Euclidean Cluster Extraction object
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.2)  # was 0.2 
    ec.set_MinClusterSize(50) # was 50 
    ec.set_MaxClusterSize(5000) # was 5000 
    ec.set_SearchMethod(kdtree) # was kdtree

    # Call the extraction function to obtain cluster indices
    cluster_indices = ec.Extract()
    return cluster_indices


def fit_line(cluster_gp_points):
    cluster_coefficients_mb = np.polyfit(cluster_gp_points['x'], cluster_gp_points['y'], deg=1)
    p_xyz = np.column_stack((cluster_gp_points['x'], cluster_gp_points['y'], cluster_gp_points['z']))
    r = np.linalg.norm(p_xyz, axis=1)

    smallest_r_index = np.argmin(r)
    largest_r_index = np.argmax(r)

    start_point = cluster_gp_points[smallest_r_index]
    end_point = cluster_gp_points[largest_r_index]

    line_points = np.hstack((start_point, end_point))
    return line_points, cluster_coefficients_mb
    

def clusters_to_line(gp_cluster_indices, gp_points):
    lines = []
    coefficients_lines = []

    P_clusters = [] # NEW!
    
    for cluster in gp_cluster_indices:
        cluster_gp_points = gp_points[cluster]
        cluster_gp_points['z'] = np.min(cluster_gp_points['z'])
        line_points, cluster_coefficients_mb = fit_line(cluster_gp_points)
        P_clusters.extend(zip(cluster_gp_points['x'], cluster_gp_points['y'], cluster_gp_points['z'],
                              cluster_gp_points['intensity'], cluster_gp_points['timestamp'], cluster_gp_points['ring']))
        lines.append(line_points)
        coefficients_lines.append(cluster_coefficients_mb)
    lines = np.hstack(lines) 
    P_clusters = np.array(P_clusters, dtype=custom_dtype) # NEW!

    return lines, coefficients_lines, P_clusters


def vertical_line_segmenting(segment_points, segmenting_type):
    # setting up and adding point_0 directly under the sensor
    point_1 = segment_points[0]
    point_0 = (0, 0, -0.715, point_1['intensity'], point_1['timestamp'], 0)
    point_0 = np.array(point_0, dtype=custom_dtype)
    segment_points = np.hstack((point_0, segment_points)) # dit ziet er goed uit nu!

    # first order points and get xyz value differences between consecutive points
    segment_points = np.sort(segment_points, order='ring')
    p_xyz = np.column_stack((segment_points['x'], segment_points['y'], segment_points['z']))
    diff_xyz = np.diff(p_xyz, axis=0)

    # Calculate distances and heights and angles between consecutive pointsdistances
    distances = np.linalg.norm(diff_xyz, axis=1)
    h_values = segment_points['z'][1:] - segment_points['z'][:-1]
    alpha_values = np.degrees(np.arcsin(h_values / (distances + 0.01))) # added this 0.01 term to prevent deviding by zero hence errors
    
    r = np.linalg.norm(p_xyz, axis=1)  # define the line-length 'r' of each point
    r_margin = 0.15 # dus lijn lengtes v/d punten moeten binnen en 15% margin van elkaar liggen | eventueel deze margin apart voor "walls" en "gp" maken!
    r_diff = np.logical_and((1-r_margin)*r[:-1] <= r[1:], r[1:] <= (1+r_margin)*r[:-1]) # je wilt de r_values binnen 20% van elkaar hebben in absolute line lengte
    r_diff_original = np.diff(r)

    # Define and apply thresholds
    if segmenting_type=="walls":
        alpha_threshold = 40 # was 3 # deze aanpassen voor walls
        mask_alpha = np.logical_or(alpha_values >= alpha_threshold, alpha_values <= -alpha_threshold) 
        mask_d = np.logical_and(0.03 < distances, r_diff)  # --- EIGEN INNOVATIE ---
        
        combined_mask = np.logical_and(mask_alpha, mask_d)
        combined_mask = np.insert(combined_mask, 0, True) # always make sure the first element (origin) of sensor is ground plane
    
        # Get gp and non-gp indices 
        gp_indices = np.where(combined_mask)[0]
        non_gp_indices = np.where(~combined_mask)[0]
        
        diff = np.diff(gp_indices) # dit berekent de verschillen tussen opeenvolgende punten
        diffs = np.sort(diff)
        gp_indices = gp_indices[1:][diff <=2] 

    elif segmenting_type=="gp":
        alpha_threshold = 3 # was 3
        h_min_threshold = 0.05

        mask_alpha = np.logical_and(alpha_values >= -alpha_threshold, alpha_values <= alpha_threshold)
        mask_h = np.logical_and(h_values >= -h_min_threshold, h_values <= h_min_threshold)
        
        # --- EIGEN INNOVATIE --- dit laat duidelijke verbetering zien tov onderstaande optie
        mask_d = np.logical_and(0.03 < distances, r_diff) # DEZE GEBRUIKTE IK, WERKT GOED! --- nu even geuncomment
        # was np.logical_and(0 < distances, r_diff) heb de 0.3 toegevoegd omdat daarmee veel noise wordt weg gefiltered
        # mask_d = r_diff_original > 0 # dit is denk ik de original geweest NU AAN HET TESTEN

        combined_mask = np.logical_and(np.logical_and(mask_alpha, mask_h), mask_d)
        combined_mask = np.insert(combined_mask, 0, True) # always make sure the first element (origin) of sensor is ground plane
    
        # Get gp and non-gp indices 
        gp_indices = np.where(combined_mask)[0]
        non_gp_indices = np.where(~combined_mask)[0]
    
    gp_segment = segment_points[gp_indices]
    if gp_segment.shape[0]>0:
        gp_segment = np.delete(gp_segment, (0), axis=0) # delete the first fakely created ring point in ring 0 which shouldnt exist
    return gp_segment


def channel_segmenter(points, segmentation_type):
    if segmentation_type=="gp":
        max_rings = 60 
    elif segmentation_type=="walls": 
        max_rings = 128
    mask_ring = points['ring'] <= max_rings 
    points = points[mask_ring]

    # only publish value beyond a 0.5m radius
    distances = np.sqrt(points['x']**2 + points['y']**2 + points['z']**2)
    points = points[distances > 1] # was 0.5
   
    gp_points = [] 
    thetas = np.arange(start=-180, stop=180, step=0.4) 
    for segment in thetas:
        segment_mask = np.logical_and(segment <= np.degrees(np.arctan2(points['y'], points['x'])), np.degrees(np.arctan2(points['y'], points['x'])) <= segment + 0.6)
        segment_points = points[segment_mask]
        
        if len(segment_points) > 0:
            segment_gp_points = vertical_line_segmenting(np.hstack(segment_points[:, None]), segmenting_type=segmentation_type) 
            gp_points.append(segment_gp_points)

    gp_points = np.hstack(gp_points)
    return gp_points


def polar_line(wall_points, lines, gp_cloud_array, P_ground):
    start_points = lines[::2]
    end_points = lines[1::2]

    min_angles = np.degrees(np.arctan2(start_points['y'], start_points['x']))
    max_angles = np.degrees(np.arctan2(end_points['y'], end_points['x']))
    segment_angles = np.concatenate((min_angles, max_angles))
    segment_angles = np.sort(segment_angles) 

    wall_point_angles = np.degrees(np.arctan2(wall_points['y'], wall_points['x']))

    wall_z_adjusted_points = []
    lowest_z_per_segment = np.array([], dtype=float)  # Initialize as an empty array
    lowest_ring_values = np.array([], dtype=int)    
    
    # STEP 1)  
    # Extract the lowest z-values, and min(ring-number) per wall segment and give all wall_points that z-value
    for i in range(len(segment_angles)-1): 
        mask_segment_points = np.logical_and(wall_point_angles > segment_angles[i], wall_point_angles <= segment_angles[i+1])
        segment_wallpoints = wall_points[mask_segment_points]
        lowest_seg_ring = np.min(segment_wallpoints['ring']) # per wall_segment slaan we de laagste ring value op: dit om start te bepalen hoe de gp te reconstructen!
        lowest_ring_values = np.append(lowest_ring_values, lowest_seg_ring)
        lowest_z_per_segment = np.append(lowest_z_per_segment, np.min(segment_wallpoints['z']))
        wall_z_adjusted_points.append(list(zip(segment_wallpoints['x'], segment_wallpoints['y'],np.full_like(segment_wallpoints['z'], fill_value=np.min(segment_wallpoints['z'])),
                                    segment_wallpoints['intensity'], segment_wallpoints['timestamp'], segment_wallpoints['ring'])))
        

    # STEP 2) 
    # For each segm. check if # gp_cloud_points > # expect gp points: if so, use those gp_points 
    skip_seg_indices = []
    new_points_list = []
    gp_point_angles = np.degrees(np.arctan2(gp_cloud_array['y'], gp_cloud_array['x']))
  
    segment_index = 0 
    for i in range(len(segment_angles) -1): 
        mask_gp_segment_points = np.logical_and(gp_point_angles > segment_angles[i], gp_point_angles <= segment_angles[i+1])
        height_mask_gp = np.logical_and(gp_cloud_array['z'] >= -1.0, gp_cloud_array['z'] < 0) # moest eigenlijk > -1 meter zijn maar heb marge genomen
        combined_mask = np.logical_and(mask_gp_segment_points, height_mask_gp)
        gp_segment_points = gp_cloud_array[combined_mask]
        alpha = abs(segment_angles[i]-segment_angles[i+1]) # in degrees
    
        ring_number = lowest_ring_values[segment_index]
        min_gp_point_thresh = ring_number*(alpha/0.4)*0.15 # was 0.15 # with 0.4 = horizontal angular res. - thresh is set to demand at least 5% of the expected points

        if len(gp_segment_points) > min_gp_point_thresh: # of gewoon een absoluut getal invullen e.g 200
            skip_seg_indices.append(segment_index)
            new_points_list.extend(zip(gp_segment_points['x'], gp_segment_points['y'], gp_segment_points['z'], 
                                        gp_segment_points['intensity'], gp_segment_points['timestamp'], gp_segment_points['ring']))
        segment_index += 1

    # REMOVING segments where the number of GP points > the minimum number of expected gp points threshold 
    # --- onderstaande lines commenten en uncommenten geeft inzicht in of the skip_indices goed gekozen zijn! --- 
    angles_to_skip = segment_angles[skip_seg_indices]# NEW !
    # print(skip_seg_indices)
    if len(skip_seg_indices)>0:
        for index in sorted(skip_seg_indices, reverse=True):
            del wall_z_adjusted_points[index]  
            
        lowest_z_per_segment = np.delete(lowest_z_per_segment, skip_seg_indices)
        lowest_ring_values = np.delete(lowest_ring_values, skip_seg_indices)
        segment_angles = np.delete(segment_angles, skip_seg_indices) # NEW !
 

    # STEP 3) 
    # Extract z-values of the adjacent wall segments below a certain global height difference from the origin else, equal z-value to lowest z-value of all segments
    origin = [0,0,-0.715]
    segment_min_adj_z = []
    for i, (segment, lowest_z) in enumerate(zip(wall_z_adjusted_points, lowest_z_per_segment)):
        if i+1 < len(wall_z_adjusted_points) and (lowest_z_per_segment[i+1] < lowest_z or lowest_z_per_segment[-(i+1)] < lowest_z) and (abs(lowest_z - origin[2]) < 0.3): # was i+1
            segment_min_adj_z.append(min(lowest_z_per_segment[i+1],lowest_z_per_segment[-(i+1)]))

        # elif abs(lowest_z - origin[2]) > 0.3: 
        else: # door dat elif verschilde de wall_z_adjusted_points and segment_min_adj_z bij mij!! Opgelost met else: maar miss nog goed naar kijken
            segment_min_adj_z.append(np.min(lowest_z_per_segment))

    # STEP 4) 
    # Replace the z-values of the lines from the wall-based segmentation with the extract adjacent z-values from step 2
    # print(segment_angles)
    combined_segments = []    
    for i, (segment_test, lowest_z_test) in enumerate(zip(wall_z_adjusted_points, segment_min_adj_z)):
        if i+1  < len(segment_min_adj_z):
            segment_test = np.array(segment_test, dtype=custom_dtype)

            # comment / uncomment below to plot or not plot the wall clusters
            combined_segments.extend(zip(segment_test['x'], segment_test['y'], np.full_like(segment_test['z'], fill_value=lowest_z_test), 
                                            segment_test['intensity'], segment_test['timestamp'], segment_test['ring']))
                        
            interpolated_seg_points = [] # reconstructed ground segments are stored here  
            p_xyz_test = np.column_stack((segment_test['x'], segment_test['y'], segment_test['z']))
            r_test = np.linalg.norm(p_xyz_test, axis=1)

            angles = np.arange(min(segment_angles[i], segment_angles[i+1]), max(segment_angles[i], segment_angles[i+1]), 0.4)
            # angles = np.arange(segment_angles[i], segment_angles[i+1], 0.4)
            # angles = create_wrapped_array(segment_angles[i], segment_angles[i+1], 0.4)           

        

            r_dist = np.linspace(0, np.percentile(r_test, 10), np.min(segment_test['ring'])) # dit was np.median
            z_values = np.linspace(-0.7, lowest_z_test, np.min(segment_test['ring']))
            
            # reconstructing the ground surface per segment if middle value is not in between the angle to skip
            if len(angles_to_skip)>0 and not min(angles_to_skip) <= np.mean(angles) <= max(angles_to_skip):    
                for r_value in r_dist: 
                    x = r_value*np.cos(np.radians(angles))
                    y = r_value*np.sin(np.radians(angles))
                    z = np.full_like(y, fill_value=z_values[i]) 
                    intensity = np.full_like(y, fill_value=10)
                    t = np.full_like(y, fill_value=10)
                    r = np.full_like(y, fill_value=10)
                    interpolated_seg_points.extend(zip(x,y,z,intensity,t,r))

            elif len(angles_to_skip)<=1:
                for r_value in r_dist: 
                    x = r_value*np.cos(np.radians(angles))
                    y = r_value*np.sin(np.radians(angles))
                    z = np.full_like(y, fill_value=z_values[i]) 
                    intensity = np.full_like(y, fill_value=10)
                    t = np.full_like(y, fill_value=10)
                    r = np.full_like(y, fill_value=10)
                    interpolated_seg_points.extend(zip(x,y,z,intensity,t,r))
            combined_segments.extend(interpolated_seg_points)
        
        # NEW !
        
        elif i+1 == len(segment_min_adj_z):
            segment_test = np.array(segment_test, dtype=custom_dtype)
            combined_segments.extend(zip(segment_test['x'], segment_test['y'], np.full_like(segment_test['z'], fill_value=lowest_z_test), 
                                            segment_test['intensity'], segment_test['timestamp'], segment_test['ring']))           
            interpolated_seg_points = [] # reconstructed ground segments are stored here  
            p_xyz_test = np.column_stack((segment_test['x'], segment_test['y'], segment_test['z']))
            r_test = np.linalg.norm(p_xyz_test, axis=1)

            # angles = np.arange(min(segment_angles[-2], segment_angles[-1]), max(segment_angles[-2], segment_angles[-1]), 0.4)
            # angles = np.arange(segment_angles[-2], segment_angles[-1], 0.4)
            angles = create_wrapped_array(min(segment_angles[-2], segment_angles[-1]), max(segment_angles[-2], segment_angles[-1]), 0.4)           

            r_dist = np.linspace(0, np.percentile(r_test, 10), np.min(segment_test['ring'])) # dit was np.median
            z_values = np.linspace(-0.7, lowest_z_test, np.min(segment_test['ring']))
            
            # reconstructing the ground surface per segment if middle value is not in between the angle to skip
            if len(angles_to_skip)>0 and not min(angles_to_skip) <= np.mean(angles) <= max(angles_to_skip):    
                for r_value in r_dist: 
                    x = r_value*np.cos(np.radians(angles))
                    y = r_value*np.sin(np.radians(angles))
                    z = np.full_like(y, fill_value=z_values[i]) 
                    intensity = np.full_like(y, fill_value=10)
                    t = np.full_like(y, fill_value=10)
                    r = np.full_like(y, fill_value=10)
                    interpolated_seg_points.extend(zip(x,y,z,intensity,t,r))

            elif len(angles_to_skip)<=1:
                for r_value in r_dist: 
                    x = r_value*np.cos(np.radians(angles))
                    y = r_value*np.sin(np.radians(angles))
                    z = np.full_like(y, fill_value=z_values[i]) 
                    intensity = np.full_like(y, fill_value=10)
                    t = np.full_like(y, fill_value=10)
                    r = np.full_like(y, fill_value=10)
                    interpolated_seg_points.extend(zip(x,y,z,intensity,t,r))
            combined_segments.extend(interpolated_seg_points)

        elif i == len(segment_min_adj_z):
            segment_test = np.array(segment_test, dtype=custom_dtype)
            combined_segments.extend(zip(segment_test['x'], segment_test['y'], np.full_like(segment_test['z'], fill_value=lowest_z_test), 
                                            segment_test['intensity'], segment_test['timestamp'], segment_test['ring']))           
            interpolated_seg_points = [] # reconstructed ground segments are stored here  
            p_xyz_test = np.column_stack((segment_test['x'], segment_test['y'], segment_test['z']))
            r_test = np.linalg.norm(p_xyz_test, axis=1)

            # in this case, wrapping of the angles is needed!
            angles = create_wrapped_array(min(segment_angles[-1], segment_angles[0]), max(segment_angles[0], segment_angles[-1]), 0.4)           
            # angles = np.arange(min(segment_angles[-1], segment_angles[0]), max(segment_angles[-1], segment_angles[0]), 0.4)
           
            r_dist = np.linspace(0, np.percentile(r_test, 10), np.min(segment_test['ring'])) # dit was np.median
            z_values = np.linspace(-0.715, lowest_z_test, np.min(segment_test['ring']))
            
            # reconstructing the ground surface per segment if middle value is not in between the angle to skip
            if len(angles_to_skip)>0 and not min(angles_to_skip) <= np.mean(angles) <= max(angles_to_skip):    
                for r_value in r_dist: 
                    x = r_value*np.cos(np.radians(angles))
                    y = r_value*np.sin(np.radians(angles))
                    z = np.full_like(y, fill_value=z_values[i]) 
                    intensity = np.full_like(y, fill_value=10)
                    t = np.full_like(y, fill_value=10)
                    r = np.full_like(y, fill_value=10)
                    interpolated_seg_points.extend(zip(x,y,z,intensity,t,r))

            elif len(angles_to_skip)<=1:
                for r_value in r_dist: 
                    x = r_value*np.cos(np.radians(angles))
                    y = r_value*np.sin(np.radians(angles))
                    z = np.full_like(y, fill_value=z_values[i]) 
                    intensity = np.full_like(y, fill_value=10)
                    t = np.full_like(y, fill_value=10)
                    r = np.full_like(y, fill_value=10)
                    interpolated_seg_points.extend(zip(x,y,z,intensity,t,r))
            combined_segments.extend(interpolated_seg_points)
        # # END NEW!

    wall_z_adjusted_points = np.array(combined_segments, dtype=custom_dtype)

    new_points_array = np.array(new_points_list, dtype=custom_dtype)
    new_points_array = np.hstack((new_points_array, wall_z_adjusted_points))
    return new_points_array 

def wrap_degrees(value):
    return (value + 180) % 360 - 180

def create_wrapped_array(start, end, step):
    # Generate an array within the specified range
    arr = np.arange(start, end + step, step)

    # Wrap values to stay within the -180 to 180 degree range
    wrapped_arr = np.vectorize(wrap_degrees)(arr)

    return wrapped_arr


def callback1(raw_cloud):
    rospy.loginfo("Processing the denoised point cloud")
    cloud_array = pointcloud2_to_array(raw_cloud)
    cloud_array = remove_zero_vecs(cloud_array)
    global gp_cloud_array
    gp_cloud_array = channel_segmenter(cloud_array, segmentation_type="gp")
    

frames = 0 
def callback2(raw_cloud):
    global frames
    rospy.loginfo("Processing the live point cloud")
    cloud_array = pointcloud2_to_array(raw_cloud)
    cloud_array = remove_zero_vecs(cloud_array)

    wall_points = channel_segmenter(cloud_array, segmentation_type="walls")
    wall_cluster_indices = cluster_planes(np.column_stack((wall_points['x'], wall_points['y'], wall_points['z'])))
    line_cloud_array, coefficients_lines, P_ground = clusters_to_line(wall_cluster_indices, wall_points) # NEW P_ground! - dit is puur ter visualisatie van de originele clusters
    combined_cloud_array = polar_line(wall_points, line_cloud_array, gp_cloud_array, P_ground) # NEW P_ground!
    
    # --- voor testen even bij onderstaande: combined_cloud array vervangen door gp_cloud_array ---
    filtered_cloud_array = remove_zero_vecs(combined_cloud_array) # combined_cloud_array
    filtered_cloud_msg = array_to_pointcloud2(filtered_cloud_array, stamp=raw_cloud.header.stamp, frame_id="PandarSwift")
    pub.publish(filtered_cloud_msg)
    frames += 1
    print(frames)


if __name__ == '__main__':
    rospy.init_node('denoising_node', anonymous=True)
    sub1 = rospy.Subscriber('live_cloud', PointCloud2, callback1) 
    sub2 = rospy.Subscriber('denoised_cloud', PointCloud2, callback2) # denoised_cloud # wass live_cloud --- reading the raw point cloud topic for data to be proccessed, was: hesai/pandar_points
    pub = rospy.Publisher('segmented_cloud', PointCloud2, queue_size=10) # this is the topic to which the processed point gets publsihed
    rate = rospy.Rate(30)  # Adjust the rate as needed
    rospy.spin()