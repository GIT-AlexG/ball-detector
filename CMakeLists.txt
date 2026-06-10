cmake_minimum_required(VERSION 3.16)
project(BallDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "H:/Programmieren/opencv/build/" CACHE PATH "OpenCV build directory")
find_package(OpenCV 5.0 REQUIRED)

# Library
add_library(ball_detector STATIC
    ball_detector.cpp
    ball_detector.hpp
)
target_include_directories(ball_detector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ball_detector PUBLIC ${OpenCV_LIBS})

# Demo executable
add_executable(ball_detector_demo
    ball_detector_demo.cpp
)
target_link_libraries(ball_detector_demo PRIVATE ball_detector)
