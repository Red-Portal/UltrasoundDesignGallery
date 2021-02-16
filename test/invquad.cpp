
/*
 * Copyright (C) 2021  Ray Kim
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <catch2/catch.hpp>
#define BLAZE_USE_DEBUG_MODE 1

#include "../src/misc/linearalgebra.hpp"

#include <limits>
#include <cmath>
#include <random>

double const catch_eps = 1e-8;

TEST_CASE("Dense inverse quadratic", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( chol = usvg::cholesky_nothrow(A).value() );

  auto x     = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});
  auto y     = blaze::solve(A, x);
  auto truth = blaze::dot(x, y);

  REQUIRE( usvg::invquad(chol, x) == Approx(truth) );
}

TEST_CASE("Diagonal inverse quadratic", "[linear algebra]")
{
  auto A = blaze::DynamicVector<double>({1, 2, 3});

  auto chol = usvg::Cholesky<usvg::DiagonalChol>();
  REQUIRE_NOTHROW( chol = usvg::cholesky_nothrow(A).value() );

  auto x     = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});
  auto y     = x / A;
  auto truth = blaze::dot(x, y);
  
  REQUIRE( usvg::invquad(chol, x) == Approx(truth) );
}
