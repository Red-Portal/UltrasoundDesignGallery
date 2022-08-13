
/*
 * Copyright (C) 2021-2022 Kyurae Kim
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

#include "catch.hpp"
#define BLAZE_USE_DEBUG_MODE 1

#include "../src/math/linearalgebra.hpp"

#include <limits>
#include <cmath>
#include <random>

double const catch_eps = 1e-8;

TEST_CASE("Dense cholesky system solve", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto chol = usdg::Cholesky<usdg::DenseChol>();
  REQUIRE_NOTHROW( chol = usdg::cholesky_nothrow(A).value() );

  auto x     = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});

  auto b      = blaze::evaluate(A * x);
  auto x_chol = usdg::solve(chol, b);

  REQUIRE( blaze::norm(x_chol - x) < catch_eps );
}

TEST_CASE("Dense LU system solve", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 2, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto lu = usdg::LU();
  REQUIRE_NOTHROW( lu = usdg::lu(A) );

  auto x     = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});

  auto b    = blaze::evaluate(A * x);
  auto x_lu = usdg::solve(lu, b);

  REQUIRE( blaze::norm(x_lu - x) < catch_eps );
}

TEST_CASE("Cholesky LU inversion", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto chol = usdg::Cholesky<usdg::DenseChol>();
  REQUIRE_NOTHROW( chol = usdg::cholesky_nothrow(A).value() );

  auto Ainv = usdg::inverse(chol);

  REQUIRE( blaze::norm(Ainv*A - blaze::IdentityMatrix<double>(3)) < catch_eps );
}
