
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

#include "../src/math/blaze.hpp"
#include "../src/math/cholesky.hpp"
#include "../src/math/linearalgebra.hpp"

TEST_CASE("Dense log determinant", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto chol = usdg::Cholesky<usdg::DenseChol>();
  REQUIRE_NOTHROW( chol = usdg::cholesky_nothrow(A).value() );
  auto truth = log(blaze::det(A));
  
  REQUIRE( usdg::logdet(chol) == usdg::Approx(truth) );
}

TEST_CASE("Diagonal log determinant", "[linear algebra]")
{
  auto A    = blaze::DynamicVector<double>({1, 2, 3});
  auto chol = usdg::Cholesky<usdg::DiagonalChol>();
  REQUIRE_NOTHROW( chol = usdg::cholesky_nothrow(A).value() );

  auto truth = log(prod(A));
  
  REQUIRE( usdg::logdet(chol) == usdg::Approx(truth) );
}

TEST_CASE("LU log determinant", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 2, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto lu = usdg::LU();
  REQUIRE_NOTHROW( lu = usdg::lu(A) );
  auto truth = log(blaze::det(A));
  
  REQUIRE( usdg::logdet(lu) == usdg::Approx(truth) );
}

TEST_CASE("LU log absolute determinant", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{0.0944588,  0.0657655, 0.787083},
     {0.747948,   0.15963,  -0.953662},
     {-0.563567, -0.384138, -0.947089}});

  auto lu = usdg::LU();
  REQUIRE_NOTHROW( lu = usdg::lu(A) );
  auto truth = log(abs(blaze::det(A)));
  
  REQUIRE( usdg::logabsdet(lu) == usdg::Approx(truth) );
}
