
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

TEST_CASE("Dense cholesky inversion", "[linear algebra]")
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

  auto b      = blaze::evaluate(A * x);
  auto x_chol = usvg::solve(chol, b);

  REQUIRE( blaze::norm(x_chol - x) < catch_eps );
}

TEST_CASE("Dense LU inversion", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 2, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto lu = usvg::LU();
  REQUIRE_NOTHROW( lu = usvg::lu(A) );

  auto x     = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});

  auto b    = blaze::evaluate(A * x);
  auto x_lu = usvg::solve(lu, b);

  REQUIRE( blaze::norm(x_lu - x) < catch_eps );
}
