
# Copyright (C) 2021  Ray Kim
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

cmake_minimum_required(VERSION 3.15)

# Set the project name to your project name, my project isn't very descriptive
project(us_gallery LANGUAGES CUDA CXX)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
  if(ENABLE_BUILD_WITH_TIME_TRACE)
    target_compile_options(project_options INTERFACE -ftime-trace)
  endif()
endif()

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# enable cache system
include(cmake/Cache.cmake)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(cmake/Sanitizers.cmake)
    enable_sanitizers(project_options)
endif()

# enable doxygen
include(cmake/Doxygen.cmake)
enable_doxygen()

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

# Release mode options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto ${CMAKE_CXX_FLAGS_RELEASE}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Rpass-analysis=loop-vectorize  -Rpass-missed=loop-vectorize")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp-simd ${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif()

# Dependencies
find_package(SFML 2.5   REQUIRED COMPONENTS system window graphics network audio )
find_package(ImGui-SFML REQUIRED)
find_package(Threads    REQUIRED)
find_package(OpenMP     REQUIRED)
find_package(Blaze      REQUIRED)
find_package(NLOPT      REQUIRED)
find_package(OpenCV     REQUIRED COMPONENTS core imgproc highgui cudafilters)

add_library(opencv INTERFACE)
target_include_directories(opencv INTERFACE "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(opencv INTERFACE "${OpenCV_LIBS}")

set(BLA_STATIC ON)
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)

add_compile_definitions(SPDLOG_FMT_EXTERNAL)
add_library(spdlog INTERFACE)
target_include_directories(spdlog SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/spdlog/include")

add_compile_definitions(FMT_HEADER_ONLY)
add_library(FMT INTERFACE)
target_include_directories(FMT SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/fmt/include")

add_library(csvparser INTERFACE)
target_include_directories(csvparser SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/csv-parser/single_include")

add_library(progressbar INTERFACE)
target_include_directories(progressbar SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/progressbar")

add_library(Random123 INTERFACE)
target_include_directories(Random123 SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/Random123")

add_library(PortableFileDialogs INTERFACE)
target_include_directories(PortableFileDialogs SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/portable-file-dialogs")

find_package(Python COMPONENTS Development NumPy REQUIRED)

find_package(CUDA REQUIRED)

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
option(ENABLE_TESTING "Enable Test Builds" ON)
option(ENABLE_FUZZING "Enable Fuzzing Builds" OFF)

if(ENABLE_TESTING)
  enable_testing()
  message("Building Tests. Be sure to check out test/constexpr_tests for constexpr testing")
  add_subdirectory(test)
endif()

# Bayesian optimization test
# add_executable(bayesopt_benchmark src/bayesopt_benchmark.cpp)
# target_link_libraries(bayesopt_benchmark
#     INTERFACE
#     blaze::blaze
#     Threads::threads

#     PUBLIC
#     Python::NumPy
#     Python::Python
#     OpenMP::OpenMP_CXX
#     csvparser
#     progressbar
#     project_options
#     project_warnings
#     BLAS::BLAS
#     LAPACK::LAPACK
#     spdlog
#     FMT
#     NLOPT::nlopt
#     Random123)


# Image processing core
add_library(ipcore
    src/imaging/pmad.cu
    src/imaging/pmad_shock.cu
    src/imaging/lpndsf.cpp)

target_link_libraries(ipcore
    INTERFACE
    project_options
    project_warnings

    PUBLIC
    opencv_core
    opencv_cudafilters
    )

set_target_properties(ipcore
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# Image processing demo

add_executable(image_processing_demo
        src/image_processing_demo.cpp)

target_link_libraries(image_processing_demo
    INTERFACE
    project_options
    project_warnings

    PUBLIC
    opencv_highgui
    ipcore)

# Ultrasound Design Gallery
add_executable(usgallery
    src/ultrasound_design_gallery.cpp
    src/ui/user_interface.cpp
    src/ui/video_player.cpp
    src/ui/line_search.cpp
    )

target_include_directories(usgallery PUBLIC ${SFML_INCLUDE_DIRS})
target_link_libraries(usgallery
    INTERFACE
    blaze::blaze
    Threads::threads

    PUBLIC
    #OpenMP::OpenMP_CXX
    opencv_highgui
    ipcore
    Python::NumPy
    Python::Python
    project_options
    project_warnings
    FMT
    spdlog
    opencv
    BLAS::BLAS
    LAPACK::LAPACK
    Random123
    PortableFileDialogs
    ImGui-SFML::ImGui-SFML)
target_compile_definitions(usgallery
    PUBLIC
    "USGALLERY_ROOT=\"${PROJECT_SOURCE_DIR}\"")
