
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

find_package(Catch2 REQUIRED)
include(CTest)
include(Catch)

add_library(catch_main STATIC catch_main.cpp)
target_link_libraries(catch_main PUBLIC Catch2::Catch2)
target_link_libraries(catch_main PRIVATE project_options)

add_executable(usgallery_tests
    statistical_test.cpp
    inverse.cpp
    invquad.cpp
    logdet.cpp
    mvnormal.cpp
    kernel.cpp
    dataset.cpp
    imh.cpp
    ess.cpp
    gp_ess.cpp
    quadrature.cpp
    laplace_bernoulli.cpp
    finitediff.cpp
    likelihood.cpp
    gp_prior.cpp
    bayesian_optimization.cpp
    )
target_link_libraries(usgallery_tests
    INTERFACE
    blaze::blaze
    Threads::threads

    PRIVATE
    Pagmo::pagmo
    #OpenMP::OpenMP_CXX
    Python::NumPy
    Python::Python
    progressbar
    BLAS::BLAS
    Random123
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
