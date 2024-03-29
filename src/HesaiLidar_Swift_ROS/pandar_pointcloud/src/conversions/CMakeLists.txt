include_directories(../taskflow)

find_package(PCL REQUIRED COMPONENTS common)

# find_package(PCL REQUIRED) # newly added now
find_package(OpenSSL REQUIRED)

message("OpenSSL include dir: ${OPENSSL_INCLUDE_DIR}")

include_directories(
	${OPENSSL_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}  # newly added now
	# ${catkin_INCLUDE_DIRS} # newly added now
)

link_directories(${PCL_LIBRARY_DIRS}) # VERY NEW
add_definitions(${PCL_DEFINITIONS}) # VERY NEW


add_executable(pandar_cloud_node cloud_node.cc convert.cc driver.cc laser_ts.cpp tcp_command_client.c util.c)
add_dependencies(pandar_cloud_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(pandar_cloud_node pandar_rawdata
					  pandar_input
					  pcap
					  ${PCL_INCLUDE_DIRS}
					  ${OPENSSL_LIBRARIES}
                      ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})
					  

add_library(pandar_cloud_nodelet cloud_nodelet.cc convert.cc driver.cc laser_ts.cpp tcp_command_client.c util.c)
add_dependencies(pandar_cloud_nodelet ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(pandar_cloud_nodelet 
					  pandar_rawdata 
					  pandar_input
					  pcap
					  ${PCL_INCLUDE_DIRS}
					  ${OPENSSL_LIBRARIES}
                      ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})

add_executable(pandar_transform_node transform_node.cc transform.cc convert.cc driver.cc laser_ts.cpp tcp_command_client.c util.c)
add_dependencies(pandar_transform_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(pandar_transform_node pandar_rawdata
						pandar_input
						pcap
						${PCL_INCLUDE_DIRS}
						${OPENSSL_LIBRARIES}
					  	${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})

add_library(pandar_transform_nodelet transform_nodelet.cc transform.cc convert.cc driver.cc laser_ts.cpp tcp_command_client.c util.c)
add_dependencies(pandar_transform_nodelet ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(pandar_transform_nodelet pandar_rawdata
						pandar_input
						pcap
						${PCL_INCLUDE_DIRS}
						${OPENSSL_LIBRARIES}
					  ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})


# NEWL added code here
add_executable(ground_reconstruction_node ground_reconstruction_node.cpp)
target_link_libraries(ground_reconstruction_node 
						${catkin_LIBRARIES} 
						${PCL_LIBRARIES})

# added new code for point cloud ALIGNMENT
add_executable(fusion_node cloud_fusion.cpp)
target_link_libraries(fusion_node 
						${catkin_LIBRARIES} 
						${PCL_LIBRARIES})
											
catkin_install_python(PROGRAMS
  live_cloud.py
  live_denoiser.py
  fast_seg.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(TARGETS 
			ground_reconstruction_node # NEWLY added
			fusion_node # added fusion node
            pandar_cloud_node
            pandar_cloud_nodelet
            pandar_transform_node
            pandar_transform_nodelet
		RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
		ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
		LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
