
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

TEST_CASE("Dense Cholesky", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( chol = usvg::cholesky_nothrow(A).value() );
  REQUIRE( blaze::norm(chol.L*blaze::trans(chol.L) - A) < catch_eps );
}

TEST_CASE("Dense Cholesky non-PD matrix", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{1, 1, 1},
     {1, 1, 1},
     {1, 1, 1}});

  auto L = std::optional<usvg::Cholesky<usvg::DenseChol>>();
  REQUIRE_NOTHROW( L = usvg::cholesky_nothrow(A) );
  REQUIRE( !L );
}

TEST_CASE("Diagonal Cholesky", "[linear algebra]")
{
  auto A    = blaze::DynamicVector<double>({3, 2, 1});
  auto chol = usvg::Cholesky<usvg::DiagonalChol>();
  REQUIRE_NOTHROW( chol = usvg::cholesky_nothrow(A).value() );
  REQUIRE( blaze::norm(chol.L*chol.L - A) < catch_eps );

  A    = blaze::DynamicVector<double>({0, 0, 0});
  chol = usvg::Cholesky<usvg::DiagonalChol>();
  REQUIRE_NOTHROW( chol = usvg::cholesky_nothrow(A).value() );
  REQUIRE( blaze::norm(chol.L*chol.L - A) < catch_eps );
}

TEST_CASE("Diagonal Cholesky non-PD matrix", "[linear algebra]")
{
  auto deps = std::numeric_limits<double>::epsilon();
  auto A    = blaze::DynamicVector<double>({0, 0, -deps});

  auto L = std::optional<usvg::Cholesky<usvg::DiagonalChol>>();
  REQUIRE_NOTHROW( L = usvg::cholesky_nothrow(A) );
  REQUIRE( !L );
}

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

TEST_CASE("Dense log determinant", "[linear algebra]")
{
  auto A = blaze::DynamicMatrix<double>(
    {{3, 1, 1},
     {1, 3, 1},
     {1, 1, 3}});

  auto chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( chol = usvg::cholesky_nothrow(A).value() );
  auto truth = log(blaze::det(A));
  
  REQUIRE( usvg::logdet(chol) == Approx(truth) );
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

TEST_CASE("Diagonal log determinant", "[linear algebra]")
{
  auto A    = blaze::DynamicVector<double>({1, 2, 3});
  auto chol = usvg::Cholesky<usvg::DiagonalChol>();
  REQUIRE_NOTHROW( chol = usvg::cholesky_nothrow(A).value() );

  auto truth = log(prod(A));
  
  REQUIRE( usvg::logdet(chol) == Approx(truth) );
}


TEST_CASE("Dense covariance multivariate normal density", "[mvnormal]")
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

  auto cov_chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( cov_chol = usvg::cholesky_nothrow(cov).value() );

  double truth_p    = 0.0069349873998044214;
  REQUIRE( usvg::dmvnormal(x, mu, cov_chol) == Approx(truth_p) );

  double truth_logp = -4.971176042116139;
  REQUIRE( usvg::dmvnormal(x, mu, cov_chol, true) == Approx(truth_logp) );
}

TEST_CASE("Diagonal covariance multivariate normal density", "[mvnormal]")
{
  auto cov = blaze::DynamicVector<double>({1.0, 2.0, 3.0});
  auto mu  = blaze::DynamicVector<double>(
    {-0.20617401141446381,
     0.15186815822664115,
     -0.03498553786495774});
  auto x  = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});

  auto cov_chol = usvg::Cholesky<usvg::DiagonalChol>();
  REQUIRE_NOTHROW( cov_chol = usvg::cholesky_nothrow(cov).value() );

  double truth_p    = 0.010371181395210441;
  REQUIRE( usvg::dmvnormal(x, mu, cov_chol) == Approx(truth_p) );

  double truth_logp = -4.568724338908423;
  REQUIRE( usvg::dmvnormal(x, mu, cov_chol, true) == Approx(truth_logp) );
}

TEST_CASE("Multivariate unit normal sampling", "[mvnormal]")
{
  auto prng        = generate_seed(1);
  size_t n_samples = 1024;
  size_t n_dims    = 16;
  auto samples     = blaze::DynamicMatrix<double>(n_dims, n_samples);

  for (size_t i = 0; i < n_samples; ++i)
  {
    blaze::column(samples, i) = usvg::rmvnormal(prng, n_dims);
  }
  auto truth_mu = 0.0;
  REQUIRE( blaze::mean(samples) == Approx(truth_mu).margin(
	     6*1/sqrt(static_cast<double>(n_samples*n_dims))) );
  auto truth_var = 1.0;
  REQUIRE( blaze::var(samples) == Approx(truth_var).margin(
	     6*1/sqrt(static_cast<double>(n_samples*n_dims))) );
}

TEST_CASE("Dense multivariate normal sampling", "[mvnormal]")
{
  auto cov = blaze::DynamicMatrix<double>(
    {{16,  1,  1},
     {1,  16,  1},
     {1,   1, 16}});
  auto mu = blaze::DynamicVector<double>(
    {1.0, 2.0, 3.0});

  auto prng     = generate_seed(1);
  auto cov_chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( cov_chol = usvg::cholesky_nothrow(cov).value() );

  size_t n_samples = 4096;
  size_t n_dims    = 3;
  auto samples     = blaze::DynamicMatrix<double>(n_dims, n_samples);
  for (size_t i = 0; i < n_samples; ++i)
  {
    blaze::column(samples, i) = usvg::rmvnormal(prng, mu, cov_chol);
  }

  /* check mean */
  size_t i = 0;
  auto est = blaze::mean(blaze::row(samples, i));
  REQUIRE( est == Approx(mu[i]).margin(
	     sqrt(cov(i,i)) / sqrt(static_cast<double>(n_samples)) * 6) );

  ++i;
  est = blaze::mean(blaze::row(samples, i));
  REQUIRE( est == Approx(mu[i]).margin(
	     sqrt(cov(i,i)) / sqrt(static_cast<double>(n_samples)) * 6) );

  ++i;
  est = blaze::mean(blaze::row(samples, i));
  REQUIRE( est == Approx(mu[i]).margin(
	     sqrt(cov(i,i)) / sqrt(static_cast<double>(n_samples)) * 6) );

  /* check var */
  i = 0;
  est = blaze::stddev(blaze::row(samples, i));
  REQUIRE( est == Approx(sqrt(cov(i,i))).margin(1.0) );

  ++i;
  est = blaze::stddev(blaze::row(samples, i));
  REQUIRE( est == Approx(sqrt(cov(i,i))).margin(1.0) );

  ++i;
  est = blaze::stddev(blaze::row(samples, i));
  REQUIRE( est == Approx(sqrt(cov(i,i))).margin(1.01) );
}


TEST_CASE("Diagonal multivariate normal sampling", "[mvnormal]")
{
  auto cov = blaze::DynamicVector<double>({16, 16, 16});
  auto mu = blaze::DynamicVector<double>({1.0, 2.0, 3.0});

  auto prng     = generate_seed(1);
  auto cov_chol = usvg::Cholesky<usvg::DiagonalChol>();
  REQUIRE_NOTHROW( cov_chol = usvg::cholesky_nothrow(cov).value() );

  size_t n_samples = 4096;
  size_t n_dims    = 3;
  auto samples     = blaze::DynamicMatrix<double>(n_dims, n_samples);
  for (size_t i = 0; i < n_samples; ++i)
  {
    blaze::column(samples, i) = usvg::rmvnormal(prng, mu, cov_chol);
  }

  /* check mean */
  size_t i = 0;
  auto est = blaze::mean(blaze::row(samples, i));
  REQUIRE( est == Approx(mu[i]).margin(
	     sqrt(cov[i]) / sqrt(static_cast<double>(n_samples)) * 6) );

  ++i;
  est = blaze::mean(blaze::row(samples, i));
  REQUIRE( est == Approx(mu[i]).margin(
	     sqrt(cov[i]) / sqrt(static_cast<double>(n_samples)) * 6) );

  ++i;
  est = blaze::mean(blaze::row(samples, i));
  REQUIRE( est == Approx(mu[i]).margin(
	     sqrt(cov[i]) / sqrt(static_cast<double>(n_samples)) * 6) );

  /* check var */
  i = 0;
  est = blaze::stddev(blaze::row(samples, i));
  REQUIRE( est == Approx(sqrt(cov[i])).margin(1.0) );

  ++i;
  est = blaze::stddev(blaze::row(samples, i));
  REQUIRE( est == Approx(sqrt(cov[i])).margin(1.0) );

  ++i;
  est = blaze::stddev(blaze::row(samples, i));
  REQUIRE( est == Approx(sqrt(cov[i])).margin(1.01) );
}