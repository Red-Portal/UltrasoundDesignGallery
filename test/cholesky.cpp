
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

#include "../src/math/cholesky.hpp"
#include "../src/math/linearalgebra.hpp"

#include <optional>

double const catch_eps = 1e-8;

TEST_CASE("Dense Cholesky", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto chol = usdg::Cholesky<usdg::DenseChol>();
  REQUIRE_NOTHROW( chol = usdg::cholesky_nothrow(A).value() );
  REQUIRE( blaze::norm(chol.L*blaze::trans(chol.L) - A) < catch_eps );
}

TEST_CASE("Dense Cholesky non-PD matrix", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{1, 1, 1},
     {1, 1, 1},
     {1, 1, 1}});

  auto L = std::optional<usdg::Cholesky<usdg::DenseChol>>();
  REQUIRE_NOTHROW( L = usdg::cholesky_nothrow(A) );
  REQUIRE( !L );
}

TEST_CASE("Diagonal Cholesky", "[linear algebra]")
{
  auto A    = blaze::DynamicVector<double>({3, 2, 1});
  auto chol = usdg::Cholesky<usdg::DiagonalChol>();
  REQUIRE_NOTHROW( chol = usdg::cholesky_nothrow(A).value() );
  REQUIRE( blaze::norm(chol.L*chol.L - A) < catch_eps );

  A    = blaze::DynamicVector<double>({0, 0, 0});
  chol = usdg::Cholesky<usdg::DiagonalChol>();
  REQUIRE_NOTHROW( chol = usdg::cholesky_nothrow(A).value() );
  REQUIRE( blaze::norm(chol.L*chol.L - A) < catch_eps );
}

TEST_CASE("Diagonal Cholesky non-PD matrix", "[linear algebra]")
{
  auto deps = std::numeric_limits<double>::epsilon();
  auto A    = blaze::DynamicVector<double>({0, 0, -deps});

  auto L = std::optional<usdg::Cholesky<usdg::DiagonalChol>>();
  REQUIRE_NOTHROW( L = usdg::cholesky_nothrow(A) );
  REQUIRE( !L );
}
