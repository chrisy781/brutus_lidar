import rospy
import numpy as np
np.float = np.float64         
from sensor_msgs.msg import PointCloud2
from ros_numpy.point_cloud2 import pointcloud2_to_array, array_to_pointcloud2
# from pygroundsegmentation import GroundPlaneFitting

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


def callback(raw_cloud):
    rospy.loginfo("Processing the point cloud")

    # Convert the raw PointCloud2 to a NumPy array
    cloud_array = pointcloud2_to_array(raw_cloud)
    cloud_array = remove_zero_vecs(cloud_array)
    # aligned_cloud_array = align_cloud(cloud_array)
    aligned_cloud_array = cloud_array[cloud_array['z']<1.8]
   
   
    # aligned_cloud_array = aligned_cloud_array[np.logical_and(aligned_cloud_array['z']<1.8,aligned_cloud_array['z']>-0.68)]
    # aligned_cloud_array = aligned_cloud_array[aligned_cloud_array['z']]

    # Convert denoised cloud for publishing
    aligned_cloud_msg = array_to_pointcloud2(aligned_cloud_array, stamp=raw_cloud.header.stamp, frame_id="PandarSwift")
    
    # Publish the denoised point cloud
    pub.publish(aligned_cloud_msg)
  

if __name__ == '__main__':
    rospy.init_node('denoising_node', anonymous=True)
    sub = rospy.Subscriber('hesai/pandar_points', PointCloud2, callback) # was: hesai/pandar_points # reading the raw point cloud topic for data to be proccessed
    pub = rospy.Publisher('live_cloud', PointCloud2, queue_size=10) # this is the topic to which the processed point gets publsihed
    rate = rospy.Rate(30)  # Adjust the rate as needed

    rospy.spin()
