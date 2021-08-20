cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(OC-SVM)

# Directory Name
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set Compiler Options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
    set(CMAKE_CXX_COMPILER "g++-8")
endif ()

# For OSX
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Find Package
find_package(Boost REQUIRED COMPONENTS program_options)

# Set Include Directories
set(INCLUDE_DIRS
    ${Boost_INCLUDE_DIRS}
    ${SRC_DIR}
)

# Set CXX_LIBRARIES
set(LIBRARIES
    ${Boost_LIBRARIES}
)

# Set Source Code
set(SRCS
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/svm.cpp
)

# Link
add_executable(${PROJECT_NAME} ${SRCS})
include_directories(${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Display Message
message(STATUS "")
message(STATUS "~~~ Boost Information ~~~")
message(STATUS "${Boost_INCLUDE_DIRS};")
message(STATUS "${Boost_LIBRARIES};")
message(STATUS "")