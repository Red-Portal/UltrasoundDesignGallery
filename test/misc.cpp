
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

#include <limits>
#include <cmath>
#include <iostream>

#include "../src/misc/linearalgebra.hpp"
#include "../src/misc/mvnormal.hpp"
#include "../src/misc/seed.hpp"

double const catch_eps = 1e-8;

TEST_CASE("Inverse quadratic", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto x     = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});
  auto y     = blaze::solve(A, x);
  auto truth = blaze::dot(x, y);

  auto L_buf = blaze::DynamicMatrix<double>(A.rows(), A.columns()); 
  blaze::llh(A, L_buf);
  auto L      = blaze::LowerMatrix<decltype(L_buf)>(blaze::decllow(L_buf));
  
  REQUIRE( usvg::invquad(L, x) == Approx(truth) );
}


TEST_CASE("Log determinant", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto L_buf = blaze::DynamicMatrix<double>(A.rows(), A.columns()); 
  blaze::llh(A, L_buf);
  auto L = blaze::LowerMatrix<decltype(L_buf)>(blaze::decllow(L_buf));

  auto truth = log(blaze::det(A));
  
  REQUIRE( usvg::logdet(L) == Approx(truth) );
}

TEST_CASE("Cholesky", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto chol = usvg::Cholesky();
  REQUIRE_NOTHROW( chol = usvg::cholesky_nothrow(A).value() );
  REQUIRE( blaze::norm(chol.L*blaze::trans(chol.L) - A) < catch_eps );
}

TEST_CASE("Cholesky Non-PD Matrix", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{1, 1, 1},
     {1, 1, 1},
     {1, 1, 1}});

  auto L = std::optional<usvg::Cholesky>();
  REQUIRE_NOTHROW( L = usvg::cholesky_nothrow(A) );
  REQUIRE( !L );
}

TEST_CASE("Multivariate Normal Density", "[mvnormal]")
{
  auto cov = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});
  auto mu = blaze::DynamicVector<double>(
    {-0.20617401141446381,
     0.15186815822664115,
     -0.03498553786495774});
  auto x  = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});

  auto cov_chol = usvg::Cholesky();
  REQUIRE_NOTHROW( cov_chol = usvg::cholesky_nothrow(cov).value() );

  double truth_p    = 0.0069349873998044214;
  REQUIRE( usvg::dmvnormal(x, mu, cov_chol) == Approx(truth_p) );

  double truth_logp = -4.971176042116139;
  REQUIRE( usvg::dmvnormal(x, mu, cov_chol, true) == Approx(truth_logp) );
}
