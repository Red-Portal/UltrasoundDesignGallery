
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
project(us_gallery CXX)
include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)

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
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# enable doxygen
include(cmake/Doxygen.cmake)
enable_doxygen()

# allow for static analysis options
include(cmake/StaticAnalyzers.cmake)

# Dependencies
find_package(SFML 2.5 COMPONENTS system window graphics network audio REQUIRED)
find_package(ImGui-SFML REQUIRED)
find_package(Blaze REQUIRED)
find_package(Threads REQUIRED)

set(BLA_STATIC ON)
find_package(BLAS REQUIRED)

add_library(gcem INTERFACE)
target_include_directories(gcem SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/gcem/include")

add_library(stats INTERFACE)
target_include_directories(stats SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/stats/include")

add_library(random123 INTERFACE)
target_include_directories(random123 SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/random123/include")

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
option(ENABLE_TESTING "Enable Test Builds" ON)
option(ENABLE_FUZZING "Enable Fuzzing Builds" OFF)

if(ENABLE_TESTING)
  enable_testing()
  message("Building Tests. Be sure to check out test/constexpr_tests for constexpr testing")
  add_subdirectory(test)
endif()

add_executable(usgallery src/main.cpp)
target_include_directories(usgallery PUBLIC ${SFML_INCLUDE_DIRS})
target_link_libraries(usgallery
    INTERFACE
    blaze::blaze
    Threads::threads

    PUBLIC
    BLAS::BLAS
    random123
    gcem
    stats
    project_options
    project_warnings
    ImGui-SFML::ImGui-SFML)
#CONAN_PKG::docopt.cpp
#CONAN_PKG::fmt
#CONAN_PKG::spdlog)
