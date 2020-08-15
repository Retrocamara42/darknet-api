cmake_minimum_required(VERSION 3.0)
set (CMAKE_CXX_STANDARD 11)

project(dark-api)

# Configure required libraries ...
find_package(Boost REQUIRED COMPONENTS system thread log program_options chrono)
find_package(Threads REQUIRED)
find_package(cpprestsdk REQUIRED)
find_package(OpenCV REQUIRED)

find_package(OpenSSL 1.0.1 REQUIRED)
set(OPENSSL_VERSION "1.0.1")
# Project construction ...

# headers search paths ...
set(DARK_API_MS_INCLUDE_DIR "./src/foundation/include")
set(OpenCV2_INCLUDE_DIR "/usr/local/include/opencv4/opencv2")
set(HEADER_SEARCH_PATHS ${DARK_API_MS_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${OPENSSL_INCLUDE_DIR} ${OpenCV2_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
# library search paths ...
set(OPENSSL_LIBS "${OPENSSL_LIBRARIES}")
set(DARKNET_LIBRARY "${PROJECT_SOURCE_DIR}/lib/libdarknet.so")

set(LIBRARIES_SEARCH_PATHS PRIVATE cpprestsdk::cpprest ${OPENSSL_LIBS} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${DARKNET_LIBRARY})

message(HEADER_SEARCH_PATHS " ${HEADER_SEARCH_PATHS}")
message(LIBRARIES_SEARCH_PATHS " ${LIBRARIES_SEARCH_PATHS}")

include_directories(${HEADER_SEARCH_PATHS})
# source files ...
add_executable(${PROJECT_NAME} ./src/api-darknet.cpp
                               ./src/api-controller.cpp
                               ./src/yolo_functions.cpp
                               ./src/foundation/network_utils.cpp
                               ./src/foundation/basic_controller.cpp)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES_SEARCH_PATHS})

