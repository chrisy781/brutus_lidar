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
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 10);
        sub_ = nh_.subscribe("hesai/pandar_points", 10, &PointCloudFilter::cloudCallback1, this); // was live_cloud 
        sub2_ = nh_.subscribe("denoised_cloud", 10, &PointCloudFilter::cloudCallback2, this);
    }

    void start() {
        std::cout << "Entering main function" << std::endl;
        ros::spin();
        std::cout << "Ran completed function" << std::endl;
    }

private:

    std::pair<std::vector<pcl::PointCloud<CustomPoint>::Ptr>, std::vector<pcl::PointCloud<CustomPoint>::Ptr>> clusters_to_info(const std::vector<pcl::PointCloud<CustomPoint>::Ptr> &clusters) {
        // Process each segment and accumulate the resulting points
        std::vector<pcl::PointCloud<CustomPoint>::Ptr> adjusted_z_clusters;
        std::vector<pcl::PointCloud<CustomPoint>::Ptr> limit_points_clusters;

        for (const auto &cluster : clusters) {
            pcl::PointCloud<CustomPoint>::Ptr adjusted_z_points(new pcl::PointCloud<CustomPoint>());
            pcl::PointCloud<CustomPoint>::Ptr limit_points(new pcl::PointCloud<CustomPoint>());

            if (cluster && cluster->size() >= 2) {
                // Get the minimum z-value in the cluster
                float minZ = cluster->points[0].z;
                for (const auto &point : cluster->points) {
                    if (point.z < minZ) {
                        minZ = point.z;
                    }
                }

                // Adjust the z-values of all points in the cluster
                for (auto &point : cluster->points) {
                    point.z = minZ;
                }

                double maxDistance = 0.0;
                double minDistance = std::numeric_limits<double>::max();
                size_t minDistanceIndex = 0;
                size_t maxDistanceIndex = 1;  // Initialize with the first two points

                // Calculate the distances using the custom function and find min/max distances
                for (size_t i = 0; i < cluster->size(); ++i) {
                    double distance = calculate_distance(cluster->points[i]);

                    if (distance > maxDistance) {
                        maxDistance = distance;
                        maxDistanceIndex = i;
                    }

                    if (distance < minDistance) {
                        minDistance = distance;
                        minDistanceIndex = i;
                    }
                }

                // Add the adjusted points to the result
                adjusted_z_points->insert(adjusted_z_points->end(), cluster->begin(), cluster->end());

                // Add the limit points to the result
                limit_points->push_back(cluster->points[minDistanceIndex]);
                limit_points->push_back(cluster->points[maxDistanceIndex]);
            }

            adjusted_z_clusters.push_back(adjusted_z_points);
            limit_points_clusters.push_back(limit_points);
        }

        return {adjusted_z_clusters, limit_points_clusters};
    }


    std::vector<pcl::PointCloud<CustomPoint>::Ptr> clusterAndPublish(const pcl::PointCloud<CustomPoint>::Ptr& cloud) {
        std::vector<pcl::PointCloud<CustomPoint>::Ptr> clusters;

        pcl::search::KdTree<CustomPoint>::Ptr tree(new pcl::search::KdTree<CustomPoint>);
        tree->setInputCloud(cloud);

        pcl::EuclideanClusterExtraction<CustomPoint> ec;
        ec.setClusterTolerance(0.15);   // was 0.15 maar in Python 0.20 dus aangepast :) Adjust this value according to your needs
        ec.setMinClusterSize(50);    // was 50  // Adjust this value according to your needs
        ec.setMaxClusterSize(5000);    // Adjust this value according to your needs
        ec.setSearchMethod(tree);

        std::vector<pcl::PointIndices> cluster_indices;
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        // Extract clusters
        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<CustomPoint>::Ptr cluster(new pcl::PointCloud<CustomPoint>);
            for (const auto& index : indices.indices) {
                cluster->push_back((*cloud)[index]);
            }
            cluster->width = cluster->size();
            cluster->height = 1;
            cluster->is_dense = true;

            clusters.push_back(cluster);
        }

        return clusters;
    }



    pcl::PointCloud<CustomPoint>::Ptr vertical_line_segmenting(const pcl::PointCloud<CustomPoint>::Ptr& segment_points, 
                                                               const std::string& segment_type) {
        pcl::PointCloud<CustomPoint>::Ptr result_cloud(new pcl::PointCloud<CustomPoint>());

        // Add a new first point to the segment_points cloud
        CustomPoint new_point;
        new_point.x = 0.0;
        new_point.y = 0.0;
        new_point.z = -0.70;
        new_point.intensity = segment_points->points[0].intensity;
        new_point.timestamp = segment_points->points[0].timestamp;
        new_point.ring = 0;
        result_cloud->points.insert(result_cloud->points.begin(), new_point);
        // result_cloud->push_back(new_point);

        double min_angle, max_angle;
        if (segment_type == "gp") {
            min_angle = -3.0;
            max_angle = 3.0;
        } 
        else if (segment_type == "walls") {
            min_angle = 60.0; // was 60
            max_angle = 120.0; // was 120
        }

        // Process each point in the segment and add points to the result_cloud
        for (int i = 0; i < segment_points->size() - 1; ++i) {
            double height_diff = segment_points->points[i + 1].z - segment_points->points[i].z;
            double distance_i = std::sqrt(segment_points->points[i].x * segment_points->points[i].x +
                                          segment_points->points[i].y * segment_points->points[i].y);
            double distance_i1 = std::sqrt(segment_points->points[i + 1].x * segment_points->points[i + 1].x +
                                           segment_points->points[i + 1].y * segment_points->points[i + 1].y);

            // Calculate elevation angle
            double angle = atan2(height_diff, distance_i1 - distance_i) * 180.0 / M_PI;

            // Check additional requirements
            if ((segment_type == "walls") 
                &&
                // (angle > -max_angle && angle < -min_angle) || (angle >= min_angle && angle <= max_angle)
                ((angle >= min_angle && angle <= max_angle) || (angle >= -max_angle && angle <= -min_angle))
                &&
                // (distance_i1 - distance_i) >= 0.03 // was commented
                //  &&
                std::abs((distance_i1 - distance_i) / distance_i) <= 0.15) // was 0.15
                //  &&
                // std::abs((segment_points->points[i + 1].ring - segment_points->points[i].ring)) <= 1) // was 1 !!
                // (segment_points->points[i + 1].ring - segment_points->points[i].ring) <= 1)
                    {result_cloud->push_back(segment_points->points[i + 1]);
            }
            else if ((segment_type == "gp")
                &&
                height_diff >= -0.05 && height_diff <= 0.05 
                &&
                (angle >= min_angle && angle <= max_angle)
                &&
                std::abs((distance_i1 - distance_i) / distance_i) <= 0.15 ){
                    result_cloud->push_back(segment_points->points[i + 1]);
                }
        }

        return result_cloud;
    }

    pcl::PointCloud<CustomPoint>::Ptr channel_segmenter(const pcl::PointCloud<CustomPoint>::Ptr& cloud_filtered, 
                                                        const std::string& segment_type) {
        const double angle_step = 0.4; // was 0.6
        const int num_segments = static_cast<int>(360.0 / angle_step);
        std::vector<pcl::PointCloud<CustomPoint>::Ptr> segment_points_vec(num_segments);

        // First do some very high-level preprocessing of the points
        pcl::PointCloud<CustomPoint>::Ptr cloud_processed(new pcl::PointCloud<CustomPoint>());

        for (const auto& point : cloud_filtered->points) {
            // Calculate distance from origin
            double distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

            if (segment_type=="gp" && distance >= 0.5 && point.ring <= 60) { // added segment_type == "gp"
                cloud_processed->push_back(point);
            }
            else if (segment_type=="walls" && distance >= 1 && point.ring <= 120) { // was 0.5 fully NEW statement 
                cloud_processed->push_back(point);
            }
        }

        // Create vertical segments of points based on 360/angular resolution
        for (const auto& point : cloud_processed->points) {
            double angle = atan2(point.y, point.x) * 180.0 / M_PI;
            int segment_index = static_cast<int>((angle + 180.0) / angle_step); // what happens here?!

            if (segment_index >= 0 && segment_index <= num_segments) { // was segment_index < num_segments
                if (!segment_points_vec[segment_index]) {
                    segment_points_vec[segment_index] = boost::make_shared<pcl::PointCloud<CustomPoint>>();
                }
                segment_points_vec[segment_index]->push_back(point); // this is a vector which contains all the points per segment, so one can iterate over segments
            }
        }

        // Process each vertical segment and accumulate the resulting points (either ground points or wall points)
        pcl::PointCloud<CustomPoint>::Ptr accumulated_cloud(new pcl::PointCloud<CustomPoint>());
        for (int i = 0; i < num_segments; ++i) {
            if (segment_points_vec[i]) {
                pcl::PointCloud<CustomPoint>::Ptr result_cloud = vertical_line_segmenting(segment_points_vec[i], segment_type);
                accumulated_cloud->insert(accumulated_cloud->end(), result_cloud->begin(), result_cloud->end());
            }
        }

        return accumulated_cloud;
    }

    pcl::PointCloud<CustomPoint>::Ptr polar_reconstruction(const std::vector<pcl::PointCloud<CustomPoint>::Ptr> &clusters_zadj,
                                                           const std::vector<pcl::PointCloud<CustomPoint>::Ptr> &limit_points, 
                                                           const pcl::PointCloud<CustomPoint>::Ptr& processed_point_cloud_) {

        pcl::PointCloud<CustomPoint>::Ptr merged_clusters(new pcl::PointCloud<CustomPoint>());
        for (size_t i = 0; i < clusters_zadj.size(); ++i) {
            *merged_clusters += *clusters_zadj[i];
        }
        
        std::vector<double> azimuth_angles; // Array to store azimuth angles
        for (double angle = 0.0; angle < 360.0; angle += 10.0) {
            azimuth_angles.push_back(angle);
        }

        // Sort the array in ascending order
        std::sort(azimuth_angles.begin(), azimuth_angles.end());

        std::vector<pcl::PointCloud<CustomPoint>::Ptr> wall_adjusted_clusters;
        std::vector<pcl::PointCloud<CustomPoint>::Ptr> gp_segments_included;
        std::vector<size_t> skip_seg_indices;
        std::vector<double> lowest_z_values;
        std::vector<int> lowest_ring_values;
        std::vector<size_t> empty_indices; // NEWLY added for reconstruction


        // Loop through each segment and extract points from merged_clusters
        for (size_t i = 0; i < azimuth_angles.size(); ++i) {  // was .size()-1
            pcl::PointCloud<CustomPoint>::Ptr segment_clusters(new pcl::PointCloud<CustomPoint>());

            if (i<azimuth_angles.size()-1){
                double angle_lower = azimuth_angles[i];
                double angle_upper = azimuth_angles[i + 1];
                 // Iterate through points in merged_clusters and add points within the angle limits to segment_clusters
                for (const auto &point : merged_clusters->points) {
                    double azimuth_point = std::atan2(point.y, point.x) * 180.0 / M_PI;
                    azimuth_point = normalizeDegrees(azimuth_point);

                    if (isAngleBetween(azimuth_point, angle_lower, angle_upper)) {
                        segment_clusters->push_back(point);
                    }
                }

            }

            else if (i==azimuth_angles.size()-1){
                double angle_lower = azimuth_angles[i];
                double angle_upper = azimuth_angles[0];
                
                for (const auto &point : merged_clusters->points) {
                    double azimuth_point = std::atan2(point.y, point.x) * 180.0 / M_PI;
                    azimuth_point = normalizeDegrees(azimuth_point);

                    // double smallest_angle = calculateSmallestAngle(angle_lower, angle_upper);

                    if (isAngleBetween(azimuth_point, angle_lower, angle_upper)) {
                        segment_clusters->push_back(point);
                    }
                }
            }
            // Add the segment_clusters to the vector of clusters
            wall_adjusted_clusters.push_back(segment_clusters);

            // Store the lowest z-value and lowest ring-value in separate arrays
            double lowest_z_value = std::numeric_limits<double>::max();
            int lowest_ring_value = std::numeric_limits<int>::max();

            for (const auto &point : segment_clusters->points) {
                if (point.z < lowest_z_value) {
                    lowest_z_value = point.z;
                }

                if (point.ring < lowest_ring_value) {
                    lowest_ring_value = point.ring;
                }
            }
           
            lowest_z_values.push_back(lowest_z_value);
            lowest_ring_values.push_back(lowest_ring_value);
        }

        // STEP 2) Make the number of ground points threshold
    
        // Calculate the angles of the ground points 
        std::vector<double> gp_angles;
        for (const auto &point : processed_point_cloud_->points) {
            double gp_angle = std::atan2(point.y, point.x) * 180.0 / M_PI;
            gp_angle = normalizeDegrees(gp_angle);

            gp_angles.push_back(gp_angle);
            
        }

        std::vector<double> thresh_to_skip;
        // next apply the number of ground tresh rule on a per segment basis
        for (size_t i = 0; i < azimuth_angles.size(); ++i) {  // was .size()-1
            pcl::PointCloud<CustomPoint>::Ptr gp_segment(new pcl::PointCloud<CustomPoint>());

            if (i<azimuth_angles.size()-1){
                double angle_lower = azimuth_angles[i];
                double angle_upper = azimuth_angles[i + 1];
                // Iterate through segmented ground points and add points within the angle limits to gp_segment
                for (size_t j = 0; j < gp_angles.size(); ++j) {
                    double angle = gp_angles[j];
                    if (isAngleBetween(angle, angle_lower, angle_upper) && processed_point_cloud_->points[j].z < 0 && processed_point_cloud_->points[j].z > -1) { 
                        // above a kind of strict filter was added to remove false positives gp points (for debugging purposes)
                        gp_segment->push_back(processed_point_cloud_->points[j]);
                    }
                }

                // Here should be the number of ground points threshold rule in the for of an if statement
                if (gp_segment->size() > (lowest_ring_values[i]*(std::abs(angle_lower - angle_upper) / 0.4)*0.50)) { // was * 0.7 maar ook 0.15
                    gp_segments_included.push_back(gp_segment);
                    skip_seg_indices.push_back(i);
                    thresh_to_skip.push_back(i);

                    }

            }

            else if (i==azimuth_angles.size()-1){
                double angle_lower = azimuth_angles[i];
                double angle_upper = azimuth_angles[0];
                // Iterate through segmented ground points and add points within the angle limits to gp_segment
                for (size_t j = 0; j < gp_angles.size(); ++j) {
                    double angle = gp_angles[j];
                    if (isAngleBetween(angle, angle_lower, angle_upper) && processed_point_cloud_->points[j].z < 0 && processed_point_cloud_->points[j].z > -1) { 
                        // above a kind of strict filter was added to remove false positives gp points (for debugging purposes)
                        gp_segment->push_back(processed_point_cloud_->points[j]);
                    }
                }

                    // Here should be the number of ground points threshold rule in the for of an if statement
                if (gp_segment->size() > (lowest_ring_values[i]*(std::abs(angle_lower - angle_upper) / 0.4)*0.50)) { // was * 0.7 maar ook 0.15
                    gp_segments_included.push_back(gp_segment);
                    skip_seg_indices.push_back(i); 
                    thresh_to_skip.push_back(i);
                    }
            }
        }
        

        // Identifying segments (based on azimuth angles) which contain too few wall points
        for (size_t i = 0; i < wall_adjusted_clusters.size(); ++i) {
            if (wall_adjusted_clusters[i]->size() < 10) { // was 2
                ROS_INFO("Wall_adjusted cluster REMOVED because too few points in the cluster");
                skip_seg_indices.push_back(i);
                empty_indices.push_back(i); // NEWLY add for empty reconstruction
            }
        }

        // making sure the skip_seg_indices only contain unique values:
        std::set<size_t> unique_skip_indices(skip_seg_indices.begin(), skip_seg_indices.end());
        skip_seg_indices.assign(unique_skip_indices.begin(), unique_skip_indices.end());

        // STEP 3) Extract z-values of adjacent z-values and more on to the reconstruction part?!

        std::vector<double> segment_min_adj_z;
        Eigen::Vector3d origin(0.0, 0.0, -0.70);

        for (size_t i = 0; i < wall_adjusted_clusters.size(); ++i) {
            
            if (wall_adjusted_clusters[i]->points.size()>2){
                std::vector<double> distances;
                for (size_t s = 0; s < wall_adjusted_clusters[i]->points.size(); ++s) {
                    distances.push_back(calculate_distance(wall_adjusted_clusters[i]->points[s]));
                }
                double median_distance = calculate_median_distance(distances);
                double elevation_angle = std::atan2(lowest_z_values[i]-origin[2],
                                                    median_distance)* 180.0 / M_PI;
                
                if (i+1<wall_adjusted_clusters.size()-1){
                    if ((lowest_z_values[i + 1] < lowest_z_values[i] || lowest_z_values[i - 1] < lowest_z_values[i]) && 
                        isAngleBetween(elevation_angle,-5,5)) { 
                        segment_min_adj_z.push_back(std::min(lowest_z_values[i + 1], lowest_z_values[i - 1])); 
                    } else {
                        segment_min_adj_z.push_back(origin[2]);
                    }
                }
                else if (i==0){
                    if ((lowest_z_values[i + 1] < lowest_z_values[i] || lowest_z_values[wall_adjusted_clusters.size()-1] < lowest_z_values[i]) && 
                        isAngleBetween(elevation_angle,-5,5)) {  
                        segment_min_adj_z.push_back(std::min(lowest_z_values[i + 1], lowest_z_values[wall_adjusted_clusters.size() - 1])); 
                    } else {
                        segment_min_adj_z.push_back(origin[2]);
                    }
                }
                else if (i==wall_adjusted_clusters.size()-1){
                    if ((lowest_z_values[0] < lowest_z_values[i] || lowest_z_values[i-1] < lowest_z_values[i]) && 
                        isAngleBetween(elevation_angle,-5,5)) { 
                        segment_min_adj_z.push_back(std::min(lowest_z_values[0], lowest_z_values[i - 1])); 
                    } else {
                        segment_min_adj_z.push_back(origin[2]);
                    }
                }
            }

            else {
                // segment_min_adj_z.push_back(origin[2]);
                if (i+1<wall_adjusted_clusters.size()-1){
                    if (lowest_z_values[i + 1] < lowest_z_values[i] || lowest_z_values[i - 1] < lowest_z_values[i]) { 
                        segment_min_adj_z.push_back(std::min(lowest_z_values[i + 1], lowest_z_values[i - 1])); 
                    } else {
                        segment_min_adj_z.push_back(origin[2]);
                    }

                    if (lowest_ring_values[i + 1] < lowest_ring_values[i] || lowest_ring_values[i - 1] < lowest_ring_values[i]) { 
                        lowest_ring_values[i] = std::min(lowest_ring_values[i + 1], lowest_ring_values[i - 1]); 
                    } 
                }
                
                else if (i==0){
                    if (lowest_z_values[i + 1] < lowest_z_values[i] || lowest_z_values[wall_adjusted_clusters.size() - 1] < lowest_z_values[i]){ 
                        segment_min_adj_z.push_back(std::min(lowest_z_values[i + 1], lowest_z_values[wall_adjusted_clusters.size() - 1])); 
                    } else {
                        segment_min_adj_z.push_back(origin[2]);
                    }

                    if (lowest_ring_values[i + 1] < lowest_ring_values[i] || lowest_ring_values[wall_adjusted_clusters.size() - 1] < lowest_ring_values[i]) { 
                        lowest_ring_values[i] = std::min(lowest_ring_values[i + 1], lowest_ring_values[wall_adjusted_clusters.size() - 1]); 
                    } 
                }

                else if (i==wall_adjusted_clusters.size()-1){
                    if (lowest_z_values[0] < lowest_z_values[i] || lowest_z_values[i - 1] < lowest_z_values[i]) { 
                        segment_min_adj_z.push_back(std::min(lowest_z_values[0], lowest_z_values[i - 1])); 
                    } else {
                        segment_min_adj_z.push_back(origin[2]);
                    }

                    if (lowest_ring_values[0] < lowest_ring_values[i] || lowest_ring_values[i - 1] < lowest_ring_values[i]) { 
                        lowest_ring_values[i] = std::min(lowest_ring_values[0], lowest_ring_values[i - 1]); 
                    } 
                }
            }
        }
     

        // STEP 4) de code voor reconstructen van points implementeren volgens de pseudo-code in m'n college blok!
        std::vector<pcl::PointCloud<CustomPoint>::Ptr> combined_segments;
        std::vector<double> min_z_vector;
        std::vector<double> min_ring_vector;
        std::vector<double> min_distance_vector;

        for (size_t i = 0; i < wall_adjusted_clusters.size(); ++i) { 
         
        
            if (i < wall_adjusted_clusters.size()-1 && !(std::find(thresh_to_skip.begin(), thresh_to_skip.end(), i) != thresh_to_skip.end() ||
                                                         std::find(empty_indices.begin(), empty_indices.end(), i) != empty_indices.end())){ 

                for (size_t v = 0; v < wall_adjusted_clusters[i]->size(); ++v) {
                    wall_adjusted_clusters[i]->points[v].z = origin[2];
                }
                combined_segments.push_back(wall_adjusted_clusters[i]);

                std::vector<double> vector_angles;
                for (double angle = std::min(azimuth_angles[i], azimuth_angles[i + 1]); angle <= std::max(azimuth_angles[i], azimuth_angles[i + 1]); angle += 0.8) {
                    vector_angles.push_back(angle);
                }

                std::vector<double> distances;
                double min_ring_value = std::numeric_limits<double>::max();
                for (size_t s = 0; s < wall_adjusted_clusters[i]->points.size(); ++s) {
                        distances.push_back(calculate_distance(wall_adjusted_clusters[i]->points[s]));
                        min_ring_value = std::min(min_ring_value, static_cast<double>(wall_adjusted_clusters[i]->points[s].ring));
                    }

                double median_distance = calculate_percentile_distance(distances, 10);
                // double ring_value = min_ring_value;
                double ring_value = lowest_ring_values[i];
                
                min_z_vector.push_back(segment_min_adj_z[i]);
                min_ring_vector.push_back(min_ring_value);
                min_distance_vector.push_back(median_distance);

                int num_steps = static_cast<int>(ring_value); 
                std::vector<double> vector_r_dist = linspace(0, median_distance, num_steps);
                std::vector<double> vector_z_values = linspace(origin[2], segment_min_adj_z[i] , num_steps); 

                pcl::PointCloud<CustomPoint>::Ptr interpolated_points(new pcl::PointCloud<CustomPoint>()); 

                auto min_angle = std::min_element(vector_angles.begin(), vector_angles.end());
                auto max_angle = std::max_element(vector_angles.begin(), vector_angles.end());

                for (size_t j = 0; j < vector_r_dist.size(); ++j) {
                    double z = vector_z_values[j];
                    double r = vector_r_dist[j];
                    for (size_t k = 0; k < vector_angles.size(); ++k) {
                        double angle_degrees = vector_angles[k];                            
                        // Map angle to the range [-180, 180)
                        if (angle_degrees >= 180.0) {
                            angle_degrees -= 360.0;
                        }

                        double angle_radians = degreesToRadians(angle_degrees);
                        double x = r * std::cos(angle_radians);
                        double y = r * std::sin(angle_radians);

                        CustomPoint interpolated_point;
                        interpolated_point.x = x;
                        interpolated_point.y = y;
                        interpolated_point.z = z;
                        interpolated_point.intensity = 10;
                        interpolated_point.timestamp = 10;
                        interpolated_point.ring = 10;

                        interpolated_points->push_back(interpolated_point);
                    }
                }
                combined_segments.push_back(interpolated_points);
            }
            
            else if (i == wall_adjusted_clusters.size()-1 && !(std::find(thresh_to_skip.begin(), thresh_to_skip.end(), i) != thresh_to_skip.end() ||
                                                        std::find(empty_indices.begin(), empty_indices.end(), i) != empty_indices.end())){ 
                for (size_t v = 0; v < wall_adjusted_clusters[i]->size(); ++v) {
                    wall_adjusted_clusters[i]->points[v].z = origin[2];
                }
                combined_segments.push_back(wall_adjusted_clusters[i]);

                std::vector<double> vector_angles;
                int num_increments = static_cast<int>(((180 - azimuth_angles[azimuth_angles.size()-1]) + (180 + azimuth_angles[0]))/0.8);
               
                for (int n = 0; n <= num_increments; ++n) { // was <=
                    double angle = azimuth_angles[azimuth_angles.size()-1] + n * 0.8;
                    if (angle >= 360.0) {
                        angle -= 360.0;
                    }
                    vector_angles.push_back(angle);
                }

                std::vector<double> distances;
                double min_ring_value = std::numeric_limits<double>::max();
                for (size_t s = 0; s < wall_adjusted_clusters[i]->points.size(); ++s) {
                        distances.push_back(calculate_distance(wall_adjusted_clusters[i]->points[s]));
                        min_ring_value = std::min(min_ring_value, static_cast<double>(wall_adjusted_clusters[i]->points[s].ring));
                    }


                double median_distance = calculate_percentile_distance(distances, 10);
                double ring_value = lowest_ring_values[i];

                min_z_vector.push_back(segment_min_adj_z[i]);
                min_ring_vector.push_back(min_ring_value);
                min_distance_vector.push_back(median_distance);

                int num_steps = static_cast<int>(ring_value); 
                std::vector<double> vector_r_dist = linspace(0, median_distance, num_steps);
                std::vector<double> vector_z_values = linspace(origin[2], segment_min_adj_z[i] , num_steps); 

                pcl::PointCloud<CustomPoint>::Ptr interpolated_points(new pcl::PointCloud<CustomPoint>()); 

                auto min_angle = std::min_element(vector_angles.begin(), vector_angles.end());
                auto max_angle = std::max_element(vector_angles.begin(), vector_angles.end());

                for (size_t j = 0; j < vector_r_dist.size(); ++j) {
                    double z = vector_z_values[j];
                    double r = vector_r_dist[j];
                    for (size_t k = 0; k < vector_angles.size(); ++k) {
                        double angle_degrees = vector_angles[k];                            
                        // Map angle to the range [-180, 180)
                        if (angle_degrees >= 180.0) {
                            angle_degrees -= 360.0;
                        }

                        double angle_radians = degreesToRadians(angle_degrees);
                        double x = r * std::cos(angle_radians);
                        double y = r * std::sin(angle_radians);

                        CustomPoint interpolated_point;
                        interpolated_point.x = x;
                        interpolated_point.y = y;
                        interpolated_point.z = z;
                        interpolated_point.intensity = 10;
                        interpolated_point.timestamp = 10;
                        interpolated_point.ring = 10;

                        interpolated_points->push_back(interpolated_point);
                    }
                }
                combined_segments.push_back(interpolated_points);
            }

            else {
                min_z_vector.push_back(4);
                min_ring_vector.push_back(60);
                min_distance_vector.push_back(4);
            }
        }


        for (size_t w = 0; w < wall_adjusted_clusters.size(); ++w) { 
    
            if (w < wall_adjusted_clusters.size()-1 && !(std::find(thresh_to_skip.begin(), thresh_to_skip.end(), w) != thresh_to_skip.end()) &&
                                                        std::find(empty_indices.begin(), empty_indices.end(), w) != empty_indices.end()){ 
                for (size_t v = 0; v < wall_adjusted_clusters[w]->size(); ++v) {
                    wall_adjusted_clusters[w]->points[v].z = origin[2];
                }
                combined_segments.push_back(wall_adjusted_clusters[w]);

                std::vector<double> vector_angles;
                for (double angle = std::min(azimuth_angles[w], azimuth_angles[w + 1]); angle <= std::max(azimuth_angles[w], azimuth_angles[w + 1]); angle += 0.8) {
                    vector_angles.push_back(angle);
                }
     
                double median_distance; 
                double ring_value;
                double min_z_value;

                auto it_median_distance = std::find_if(min_distance_vector.begin(), min_distance_vector.end(), [](double val) {
                    return val < 4.0;
                });
                median_distance = (it_median_distance != min_distance_vector.end()) ? *it_median_distance : *std::min_element(min_distance_vector.begin(), min_distance_vector.end());

                auto it_ring_value = std::find_if(min_ring_vector.begin(), min_ring_vector.end(), [](double val) {
                    return val < 60.0;
                });
                ring_value = (it_ring_value != min_ring_vector.end()) ? *it_ring_value : *std::min_element(min_ring_vector.begin(), min_ring_vector.end());

                auto it_min_z_value = std::find_if(min_z_vector.begin(), min_z_vector.end(), [](double val) {
                    return val < 4.0;
                });
                min_z_value = (it_min_z_value != min_z_vector.end()) ? *it_min_z_value : *std::min_element(min_z_vector.begin(), min_z_vector.end());

                // std::cout << "median disance: " << median_distance << std::endl;
                // std::cout << "ring value: " << ring_value << std::endl;
                // std::cout << "min z value: " << min_z_value << std::endl;

                int num_steps = static_cast<int>(ring_value); 
                std::vector<double> vector_r_dist = linspace(0, median_distance, num_steps);
                std::vector<double> vector_z_values = linspace(origin[2], min_z_value , num_steps); 

                pcl::PointCloud<CustomPoint>::Ptr interpolated_points(new pcl::PointCloud<CustomPoint>()); 

                auto min_angle = std::min_element(vector_angles.begin(), vector_angles.end());
                auto max_angle = std::max_element(vector_angles.begin(), vector_angles.end());

                for (size_t j = 0; j < vector_r_dist.size(); ++j) {
                    double z = vector_z_values[j];
                    double r = vector_r_dist[j];
                    for (size_t k = 0; k < vector_angles.size(); ++k) {
                        double angle_degrees = vector_angles[k];                            
                        // Map angle to the range [-180, 180)
                        if (angle_degrees >= 180.0) {
                            angle_degrees -= 360.0;
                        }

                        double angle_radians = degreesToRadians(angle_degrees);
                        double x = r * std::cos(angle_radians);
                        double y = r * std::sin(angle_radians);

                        CustomPoint interpolated_point;
                        interpolated_point.x = x;
                        interpolated_point.y = y;
                        interpolated_point.z = z;
                        interpolated_point.intensity = 10;
                        interpolated_point.timestamp = 10;
                        interpolated_point.ring = 10;

                        interpolated_points->push_back(interpolated_point);
                    }
                }
                combined_segments.push_back(interpolated_points);
            }

            else if (w == wall_adjusted_clusters.size()-1 && !(std::find(thresh_to_skip.begin(), thresh_to_skip.end(), w) != thresh_to_skip.end()) &&
                                                            std::find(empty_indices.begin(), empty_indices.end(), w) != empty_indices.end()){ 
                for (size_t v = 0; v < wall_adjusted_clusters[w]->size(); ++v) {
                    wall_adjusted_clusters[w]->points[v].z = origin[2];
                }
                combined_segments.push_back(wall_adjusted_clusters[w]);

                std::vector<double> vector_angles;
                int num_increments = static_cast<int>(((180 - azimuth_angles[azimuth_angles.size()-1]) + (180 + azimuth_angles[0]))/0.8);
               
                for (int n = 0; n <= num_increments; ++n) { // was <=
                    double angle = azimuth_angles[azimuth_angles.size()-1] + n * 0.8;
                    if (angle >= 360.0) {
                        angle -= 360.0;
                    }
                    vector_angles.push_back(angle);
                }

                double median_distance; 
                double ring_value; 
                double min_z_value;

                auto it_median_distance = std::find_if(min_distance_vector.begin(), min_distance_vector.end(), [](double val) {
                    return val < 4.0;
                });
                median_distance = (it_median_distance != min_distance_vector.end()) ? *it_median_distance : *std::min_element(min_distance_vector.begin(), min_distance_vector.end());

                auto it_ring_value = std::find_if(min_ring_vector.begin(), min_ring_vector.end(), [](double val) {
                    return val < 60.0;
                });
                ring_value = (it_ring_value != min_ring_vector.end()) ? *it_ring_value : *std::min_element(min_ring_vector.begin(), min_ring_vector.end());

                auto it_min_z_value = std::find_if(min_z_vector.begin(), min_z_vector.end(), [](double val) {
                    return val < 4.0;
                });
                min_z_value = (it_min_z_value != min_z_vector.end()) ? *it_min_z_value : *std::min_element(min_z_vector.begin(), min_z_vector.end());

                // std::cout << "median disance: " << median_distance << std::endl;
                // std::cout << "ring value: " << ring_value << std::endl;
                // std::cout << "min z value: " << min_z_value << std::endl;

                int num_steps = static_cast<int>(ring_value); // I dont think this is needed since I found out this did not cause the segmentation error
                std::vector<double> vector_r_dist = linspace(0, median_distance, num_steps);
                std::vector<double> vector_z_values = linspace(origin[2], min_z_value , num_steps); // segment_min_adj_z[w] 

                pcl::PointCloud<CustomPoint>::Ptr interpolated_points(new pcl::PointCloud<CustomPoint>()); 

                auto min_angle = std::min_element(vector_angles.begin(), vector_angles.end());
                auto max_angle = std::max_element(vector_angles.begin(), vector_angles.end());

                for (size_t j = 0; j < vector_r_dist.size(); ++j) {
                    double z = vector_z_values[j];
                    double r = vector_r_dist[j];
                    for (size_t k = 0; k < vector_angles.size(); ++k) {
                        double angle_degrees = vector_angles[k];                            
                        // Map angle to the range [-180, 180)
                        if (angle_degrees >= 180.0) {
                            angle_degrees -= 360.0;
                        }

                        double angle_radians = degreesToRadians(angle_degrees);
                        double x = r * std::cos(angle_radians);
                        double y = r * std::sin(angle_radians);

                        CustomPoint interpolated_point;
                        interpolated_point.x = x;
                        interpolated_point.y = y;
                        interpolated_point.z = z;
                        interpolated_point.intensity = 10;
                        interpolated_point.timestamp = 10;
                        interpolated_point.ring = 10;

                        interpolated_points->push_back(interpolated_point);
                    }
                }
                combined_segments.push_back(interpolated_points);
            }
        }

        // STEP 5) Concatenate/merge the resulting combined_segments with gp_segments_included
        pcl::PointCloud<CustomPoint>::Ptr merged_cloud(new pcl::PointCloud<CustomPoint>());
        for (size_t i = 0; i < gp_segments_included.size(); ++i) {
            *merged_cloud += *gp_segments_included[i];
        }

        for (size_t i = 0; i < combined_segments.size(); ++i) {
            *merged_cloud += *combined_segments[i];
        }

        return merged_cloud;
    }

    double calculate_distance(const CustomPoint& point) {
        return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    }

    double calculate_median_distance(const std::vector<double>& distances) {
        // Sort the distances
        std::vector<double> sorted_distances(distances);
        std::sort(sorted_distances.begin(), sorted_distances.end());

        size_t size = sorted_distances.size();
     
        return (size % 2 == 0) ? 0.5 * (sorted_distances[size / 2 - 1] + sorted_distances[size / 2]) : sorted_distances[size / 2];
    }

    double calculate_percentile_distance(const std::vector<double>& distances, double percentile) {
        // Ensure the percentile is within [0, 100]
        percentile = std::max(0.0, std::min(100.0, percentile));

        // Sort the distances
        std::vector<double> sorted_distances(distances);
        std::sort(sorted_distances.begin(), sorted_distances.end());

        // Calculate the index corresponding to the percentile
        size_t size = sorted_distances.size();
        size_t index = static_cast<size_t>((percentile / 100.0) * (size - 1));

        // Interpolate if the index is not an integer
        if (index != percentile / 100.0 * (size - 1)) {
            double fraction = percentile / 100.0 * (size - 1) - index;
            return (1.0 - fraction) * sorted_distances[index] + fraction * sorted_distances[index + 1];
        } else {
            return sorted_distances[index];
        }
    }

    std::vector<double> linspace(double start, double end, size_t num_points) {
        std::vector<double> result;
        for (size_t i = 0; i < num_points; ++i) {
            double value = start + (static_cast<double>(i) / static_cast<double>(num_points - 1)) * ((end+0.005) - start);
            result.push_back(value);

        }

        return result;
    }

    double calculateSmallestAngle(double angle1, double angle2) {
        double smallestAngle = std::fmod((angle2 - angle1 + 360.0), 360.0);
        return (smallestAngle > 180.0) ? 360.0 - smallestAngle : smallestAngle;
    }

    bool isAngleBetween(double smallestAngle, double angleA, double angleB) {
        // Normalize angles to the range [0, 360)
        // angleA = normalizeDegrees(angleA);
        // angleB = normalizeDegrees(angleB);

        if (angleA < angleB) {
            return angleA <= smallestAngle && smallestAngle <= angleB;
        } else {
            return angleA <= smallestAngle || smallestAngle <= angleB;
        }
    }


    // double circularMean(double angle1, double angle2) {
    //     // Convert angles to radians
    //     double radian1 = angle1 * M_PI / 180.0;
    //     double radian2 = angle2 * M_PI / 180.0;

    //     // Calculate circular mean
    //     double meanRadian = atan2(sin(radian1) + sin(radian2), cos(radian1) + cos(radian2));

    //     // Convert mean back to degrees
    //     double meanDegree = meanRadian * 180.0 / M_PI;

    //     // Ensure the result is in the range [0, 360)
    //     if (meanDegree < 0) {
    //         meanDegree += 360.0;
    //     }

    //     return meanDegree;
    // }


    double normalizeDegrees(double degrees) {
        double normalized = std::fmod(degrees, 360.0);
        if (normalized < 0.0) {
            normalized += 360.0;
        }
        return normalized;
    }

    double degreesToRadians(double degrees) {
        return degrees * M_PI / 180.0;
    }

    double radiansToDegrees(double radians) {
        return radians * 180.0 / M_PI;
    }

    double circularMedian(const std::vector<double>& angles) {
        if (angles.empty()) {
            std::cerr << "Error: Cannot calculate circular median for an empty vector." << std::endl;
            return 0.0;  // Return 0 as a default value or handle the error as needed.
        }

        // Convert angles to unit vectors on the unit circle
        std::vector<double> unitVectorsX;
        std::vector<double> unitVectorsY;
        for (const auto& angle : angles) {
            double radianAngle = degreesToRadians(angle);
            unitVectorsX.push_back(cos(radianAngle));
            unitVectorsY.push_back(sin(radianAngle));
        }

        // Calculate mean of unit vectors
        double meanX = 0.0;
        double meanY = 0.0;
        for (size_t i = 0; i < unitVectorsX.size(); ++i) {
            meanX += unitVectorsX[i];
            meanY += unitVectorsY[i];
        }
        meanX /= unitVectorsX.size();
        meanY /= unitVectorsY.size();

        // Convert mean vector back to an angle
        double meanAngleRadians = atan2(meanY, meanX);
        double circularMedianAngle = radiansToDegrees(meanAngleRadians);

        // Ensure the result is in the range [0, 360)
        if (circularMedianAngle < 0.0) {
            circularMedianAngle += 360.0;
        }

        return circularMedianAngle;
    }

    double circularMean(double angle1, double angle2) {
        // Convert angles to radians
        double radian1 = angle1 * M_PI / 180.0;
        double radian2 = angle2 * M_PI / 180.0;

        // Calculate circular mean
        double meanRadian = atan2(sin(radian1) + sin(radian2), cos(radian1) + cos(radian2));

        // Convert mean back to degrees
        double meanDegree = meanRadian * 180.0 / M_PI;

        // // Ensure the result is in the range [-180, 180)
        // if (meanDegree < -180.0) {
        //     meanDegree += 360.0;
        // } else if (meanDegree >= 180.0) {
        //     meanDegree -= 360.0;
        // }

        // Ensure the result is in the range [-180, 180) --- the code below give the output in the 0, 360 degree range
        if (meanDegree < -180.0) {
            meanDegree += 360.0;
        } else if (meanDegree >= 180.0) {
            meanDegree -= 360.0;
        }

        // Ensure the result is in the range [0, 360)
        if (meanDegree < 0.0) {
            meanDegree += 360.0;
        }

        return meanDegree;
    }

    
    void cloudCallback1(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        pcl::PointCloud<CustomPoint>::Ptr cloud(new pcl::PointCloud<CustomPoint>());
        pcl::fromROSMsg(*cloud_msg, *cloud);
        pcl::PointCloud<CustomPoint>::Ptr accumulated_cloud = channel_segmenter(cloud, "gp"); // was: cloud_filtered,
       
        processed_point_cloud_ = accumulated_cloud;
    }

    double frames = 0; 
    void cloudCallback2(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
        sensor_msgs::PointCloud2 output_cloud_msg;

        pcl::PointCloud<CustomPoint>::Ptr cloud(new pcl::PointCloud<CustomPoint>());
        pcl::fromROSMsg(*cloud_msg, *cloud);

        pcl::PointCloud<CustomPoint>::Ptr accumulated_cloud = channel_segmenter(cloud, "walls");
        std::vector<pcl::PointCloud<CustomPoint>::Ptr> clusters = clusterAndPublish(accumulated_cloud); //accumulated_cloud
    
        std::pair<std::vector<pcl::PointCloud<CustomPoint>::Ptr>, std::vector<pcl::PointCloud<CustomPoint>::Ptr>> result = clusters_to_info(clusters);
        std::vector<pcl::PointCloud<CustomPoint>::Ptr> clusters_zadj = result.first;
        std::vector<pcl::PointCloud<CustomPoint>::Ptr> limit_points_clusters = result.second;


        // pcl::PointCloud<CustomPoint>::Ptr test_cloud(new pcl::PointCloud<CustomPoint>());
        // for (size_t i = 0; i < clusters_zadj.size(); ++i) {
        //     *test_cloud += *clusters_zadj[i];
        // }

        pcl::PointCloud<CustomPoint>::Ptr test_cloud = polar_reconstruction(clusters_zadj, limit_points_clusters, processed_point_cloud_);
        pcl::toROSMsg(*test_cloud, output_cloud_msg);
        output_cloud_msg.header = cloud_msg->header;
        pub_.publish(output_cloud_msg);
        frames += 1;
        std::cout << "Number of published frames: "<< frames << std::endl;

    }

    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;
    ros::Subscriber sub2_;
    sensor_msgs::PointCloud2 output_cloud_msg;
    pcl::PointCloud<CustomPoint>::Ptr processed_point_cloud_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ground_reconstruction_node"); // was pointcloud_filter_node
    ros::NodeHandle nh;

    PointCloudFilter filter(nh);
    filter.start();
    return 0;
}
