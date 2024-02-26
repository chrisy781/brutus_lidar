#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <boost/make_shared.hpp>
#include <cmath>
#include <vector>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h> 
#include <algorithm>
#include <std_msgs/String.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>

struct CustomPoint {
    PCL_ADD_POINT4D; // Adds X, Y, Z, and padding
    std::uint8_t intensity;
    double timestamp; // Timestamp
    std::uint16_t ring; // Ring information
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Ensures proper memory alignment
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(CustomPoint,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (std::uint8_t, intensity, intensity)
                                  (double, timestamp, timestamp)
                                  (std::uint16_t, ring, ring))

class PointCloudFilter {
public:
    PointCloudFilter(ros::NodeHandle& nh) : nh_(nh) {
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("fused_cloud", 10);
        sub_ = nh_.subscribe("/pointcloud1", 10, &PointCloudFilter::cloudCallback1, this);
        sub2_ = nh_.subscribe("/pointcloud2", 10, &PointCloudFilter::cloudCallback2, this);
        sub3_ = nh_.subscribe("/pointcloud3", 10, &PointCloudFilter::cloudCallback3, this);
        sub4_ = nh_.subscribe("/pointcloud4", 10, &PointCloudFilter::cloudCallback4, this);
    }

    void start() {
        std::cout << "Entering main function" << std::endl;
        ros::spin();
        std::cout << "Ran completed function" << std::endl;
    }

private:
 
    void cloudCallback1(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        pcl::PointCloud<CustomPoint>::Ptr cloud(new pcl::PointCloud<CustomPoint>());
        pcl::fromROSMsg(*cloud_msg, *cloud);
       
        cloud1 = cloud;
    }

    void cloudCallback2(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        pcl::PointCloud<CustomPoint>::Ptr cloud(new pcl::PointCloud<CustomPoint>());
        pcl::fromROSMsg(*cloud_msg, *cloud);
          
        cloud2 = cloud;
    }

    void cloudCallback3(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        pcl::PointCloud<CustomPoint>::Ptr cloud(new pcl::PointCloud<CustomPoint>());
        pcl::fromROSMsg(*cloud_msg, *cloud);

        cloud3 = cloud;
    }

    void cloudCallback4(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        pcl::PointCloud<CustomPoint>::Ptr cloud4(new pcl::PointCloud<CustomPoint>());
        pcl::fromROSMsg(*cloud_msg, *cloud4);
        
        
        pcl::PointCloud<CustomPoint>::Ptr transformed_cloud2(new pcl::PointCloud<CustomPoint>());
        customTransform(0.14,-3.67, 0, cloud2, transformed_cloud2, 0, 0, 2);

        pcl::PointCloud<CustomPoint>::Ptr transformed_cloud3(new pcl::PointCloud<CustomPoint>());
        customTransform(3.5,-3.2,0.02, cloud3, transformed_cloud3, 0, 1, 0);

        pcl::PointCloud<CustomPoint>::Ptr transformed_cloud4(new pcl::PointCloud<CustomPoint>());
        customTransform(2.9,0.13,0, cloud4, transformed_cloud4, -2, 0, 5);

        pcl::PointCloud<CustomPoint>::Ptr output_cloud(new pcl::PointCloud<CustomPoint>());
        *output_cloud = *cloud1 + *transformed_cloud2 + *transformed_cloud3 + *transformed_cloud4;//*transformed_cloud3; // *transformed_cloud2; //+ *transformed_cloud3 + *transformed_cloud4; // beginnen met cloud1 en cloud2 op te lijnen!

        // pcl::PassThrough<CustomPoint> pass;
        // pass.setInputCloud(output_cloud);
        // pass.setFilterFieldName("z");
        // pass.setFilterLimits(-0.53, -0.47);
        // pass.filter(*output_cloud);

        // pcl::PointCloud<CustomPoint>::Ptr downsampled_output_cloud(new pcl::PointCloud<CustomPoint>());
        // downsamplePointCloud(output_cloud, downsampled_output_cloud);

        // Publish the fused point cloud
        sensor_msgs::PointCloud2 output_cloud_msg;
        pcl::toROSMsg(*output_cloud, output_cloud_msg);
        output_cloud_msg.header = cloud_msg->header;
        pub_.publish(output_cloud_msg);
    }

    void customTransform(float translation_x,
                         float translation_y,
                         float translation_z,
                         const pcl::PointCloud<CustomPoint>::Ptr& input_cloud,
                         pcl::PointCloud<CustomPoint>::Ptr& output_cloud,
                         float rotation_angle_x,
                         float rotation_angle_y,
                         float rotation_angle_z) {


    float rotation_x_rad = rotation_angle_x * M_PI / 180.0;
    float rotation_y_rad = rotation_angle_y * M_PI / 180.0;
    float rotation_z_rad = rotation_angle_z * M_PI / 180.0;


    // Example: Translate along the x-axis and rotate around the z, x, and y axes
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << translation_x, translation_y, translation_z;  // Translation along x-axis y moest was minder negatief, x was best goed!

    transform.rotate(Eigen::AngleAxisf(rotation_x_rad, Eigen::Vector3f::UnitX()));  // Rotation around x-axis
    transform.rotate(Eigen::AngleAxisf(rotation_y_rad, Eigen::Vector3f::UnitY()));  // Rotation around y-axis
    transform.rotate(Eigen::AngleAxisf(rotation_z_rad, Eigen::Vector3f::UnitZ()));  // Rotation around z-axis

    pcl::transformPointCloud(*input_cloud, *output_cloud, transform);
    }

    void downsamplePointCloud(const pcl::PointCloud<CustomPoint>::Ptr& input_cloud,
                              pcl::PointCloud<CustomPoint>::Ptr& downsampled_cloud) {
        pcl::VoxelGrid<CustomPoint> sor;
        sor.setInputCloud(input_cloud);
        sor.setLeafSize(0.01, 0.01, 0.01);  // Adjust leaf size as needed
        sor.filter(*downsampled_cloud);
    }
    

    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;
    ros::Subscriber sub2_;
    ros::Subscriber sub3_;
    ros::Subscriber sub4_;
    sensor_msgs::PointCloud2 output_cloud_msg;
    pcl::PointCloud<CustomPoint>::Ptr cloud1;
    pcl::PointCloud<CustomPoint>::Ptr cloud2;
    pcl::PointCloud<CustomPoint>::Ptr cloud3;
    // pcl::PointCloud<CustomPoint>::Ptr aligned_cloud2;


};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_fusion_node");
    ros::NodeHandle nh;

    PointCloudFilter filter(nh);
    filter.start();
    return 0;
}
