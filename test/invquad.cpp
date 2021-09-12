
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

#include <catch2/catch_all.hpp>
#define BLAZE_USE_DEBUG_MODE 1

#include "../src/math/linearalgebra.hpp"
#include "../src/math/prng.hpp"
#include "utils.hpp"

#include <limits>
#include <cmath>
#include <random>

TEST_CASE("Dense inverse quadratic", "[linear algebra]")
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
  auto y     = blaze::solve(A, x);
  auto truth = blaze::dot(x, y);

  REQUIRE( usdg::invquad(chol, x) == Catch::Approx(truth) );
}

TEST_CASE("dense batch inverse quadratic", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto chol = usdg::Cholesky<usdg::DenseChol>();
  REQUIRE_NOTHROW( chol = usdg::cholesky_nothrow(A).value() );

  auto key  = GENERATE(range(0u, 8u));
  auto prng = usdg::Random123(key);

  auto X  = generate_mvsamples(prng, 3, 6);
  auto ys = usdg::invquad_batch(chol, X);

  for (size_t i = 0; i < 6; ++i)
  {
    REQUIRE( ys[i] == Catch::Approx(usdg::invquad(chol, blaze::column(X, i))) );
  }
}

TEST_CASE("diagonal inverse quadratic", "[linear algebra]")
{
  auto A = blaze::DynamicVector<double>({1, 2, 3});

  auto chol = usdg::Cholesky<usdg::DiagonalChol>();
  REQUIRE_NOTHROW( chol = usdg::cholesky_nothrow(A).value() );

  auto x     = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});
  auto y     = x / A;
  auto truth = blaze::dot(x, y);
  
  REQUIRE( usdg::invquad(chol, x) == Catch::Approx(truth) );
}

// TEST_CASE("Laplace approximation inverse quadratic", "[linear algebra]")
// {
//   auto Kinv = blaze::DynamicMatrix<double>(
//     {{3, 1, 1},
//      {1, 3, 1},
//      {1, 1, 3}});
//   auto W = blaze::DynamicMatrix<double>(
//     {{5, 2, 3},
//      {2, 4, 2},
//      {3, 2, 4}});
//   auto x  = blaze::DynamicVector<double>(
//     {0.9040983839157295,
//      -0.29874050736604413,
//      -1.2570687585683156});

//   auto K = blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>(Kinv);
//   blaze::invert(K);

//   auto WK    = W*K;
//   auto IpWK  = usvg::lu(blaze::IdentityMatrix<double>(W.rows()) + WK);
//   auto truth = blaze::dot(x, blaze::solve(Kinv + W, x));
//   REQUIRE( usvg::invquad(IpWK, K, WK, x) == Catch::Approx(truth) );
// }
