cmake_minimum_required(VERSION 2.8.3)
project(stereo_depth_fusion)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++11 -Wall -msse4.2")

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

find_package(OpenMP)
IF(OPENMP_FOUND)
    MESSAGE("OpenMP Enabled")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF(OPENMP_FOUND)

include_directories(include)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})
link_directories(${PNG_LIBRARY_DIRS})

message(STATUS ${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_LIBRARY_DIRS})
message(STATUS ${OpenCV_LIBS})

add_executable(stereo_depth_fusion src/stereo_depth_fusion.cpp
                                   src/SGMStereo.cpp
                                   src/utils.cpp)
target_link_libraries(stereo_depth_fusion ${OpenCV_LIBS} ${PNG_LIBS})

