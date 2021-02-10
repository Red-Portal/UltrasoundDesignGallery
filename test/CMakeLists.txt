
find_package(Catch2 REQUIRED)
include(CTest)
include(Catch)

#add_library(gcem INTERFACE)
#target_include_directories(gcem SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/gcem/include")

#add_library(stats INTERFACE)
#target_include_directories(stats SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/stats/include")

#add_library(random123 INTERFACE)
#target_include_directories(random123 SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/third_party/random123/include")

add_library(catch_main STATIC catch_main.cpp)
target_link_libraries(catch_main PUBLIC Catch2::Catch2)
target_link_libraries(catch_main PRIVATE project_options)

add_executable(usgallery_tests
    gp.cpp
    inference.cpp)
target_link_libraries(usgallery_tests
    INTERFACE
    blaze::blaze
    Threads::threads
    
    PRIVATE
    gcem
    stats
    random123
    project_warnings
    project_options catch_main)

# automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
# to whatever you want, or use different for different binaries
catch_discover_tests(
    usgallery_tests
    TEST_PREFIX
    "unittests."
    REPORTER
    xml
    OUTPUT_DIR
    .
    OUTPUT_PREFIX
    "unittests."
    OUTPUT_SUFFIX
    .xml)
