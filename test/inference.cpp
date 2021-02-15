
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

#include "../src/inference/imh.hpp"
#include "../src/misc/linearalgebra.hpp"
#include "../src/misc/mvnormal.hpp"
#include "../src/misc/prng.hpp"
#include "statistical_test.hpp"
#include "elliptical_slice.hpp"

TEST_CASE("Sampling from unit Gaussian", "[imh]")
{
  auto key  = GENERATE(range(0u, 8u));
  auto prng = usvg::Random123(key);
  auto p    = [](double x){
    return exp(-x*x/2);
  };
  size_t n_samples = 512;
  auto samples = usvg::imh(prng, p, -3, 3, n_samples, 128, 1);

  REQUIRE( !kolmogorov_smirnoff_test(0.001, normal_cdf,
				     samples.begin(), samples.end()) );
}

TEST_CASE("Dense prior elliptical slice sampling", "[ess]")
{
  auto key         = GENERATE(range(0u, 8u));
  auto prng        = usvg::Random123(key);
  size_t n_samples = 512;
  size_t n_burn    = 128;
  auto like_mean   = blaze::DynamicVector<double>(
    {1.0, 2.0, 3.0});
  auto like_cov    = blaze::DynamicMatrix<double>(
    {{  1,  0.1,  0.1},
     {0.1,    1,  0.1},
     {0.1,  0.1,    1}});
  auto like_chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( like_chol = usvg::cholesky_nothrow(like_cov).value() );
  auto like_dist  = usvg::MvNormal<usvg::DenseChol>{like_mean,  like_chol};

  auto prior_mean = blaze::DynamicVector<double>(
    {1.0, 1.0, 1.0});
  auto prior_cov  = blaze::DynamicMatrix<double>(
    {{16,  1,   1},
     {1,  16,   1},
     {1,   1,  16}});
  auto prior_chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( prior_chol = usvg::cholesky_nothrow(prior_cov).value() );
  auto prior_dist = usvg::MvNormal<usvg::DenseChol>{prior_mean, prior_chol};

  auto loglike = [&](blaze::DynamicVector<double> const& x){
    return usvg::invquad(like_dist.cov_chol, x - like_mean)/-2;
  };

  auto x0      = prior_dist.sample(prng); 
  auto samples = elliptical_slice(prng, n_samples, n_burn, x0, loglike, prior_dist);

  auto like_inv  = blaze::inv(like_cov);
  auto prior_inv = blaze::inv(prior_cov);
  auto post_cov  = blaze::evaluate(blaze::inv(like_inv + prior_inv));
  auto post_mean = blaze::evaluate(post_cov*(like_inv*like_mean + prior_inv*prior_mean));

  size_t i = 0;
  auto row = blaze::row(samples, i);
  auto cdf = [&](double x_in){
    return normal_cdf((x_in -  post_mean[i]) / sqrt(post_cov(i,i)));
  };
  REQUIRE( !kolmogorov_smirnoff_test(0.001, cdf, row.begin(), row.end()) );

  ++i;
  row = blaze::row(samples, i);
  REQUIRE( !kolmogorov_smirnoff_test(0.001, cdf, row.begin(), row.end()) );

  ++i;
  row = blaze::row(samples, i);
  REQUIRE( !kolmogorov_smirnoff_test(0.001, cdf, row.begin(), row.end()) );
}

TEST_CASE("Diagonal prior elliptical slice sampling", "[ess]")
{
  auto key         = GENERATE(range(0u, 8u));
  auto prng        = usvg::Random123(key);
  size_t n_samples = 512;
  size_t n_burn    = 512;

  auto like_mean   = blaze::DynamicVector<double>({1.0, 2.0, 3.0});
  auto like_cov    = blaze::DynamicVector<double>({ 1,  1,  1});
  auto like_chol = usvg::Cholesky<usvg::DiagonalChol>();
  REQUIRE_NOTHROW( like_chol = usvg::cholesky_nothrow(like_cov).value() );
  auto like_dist  = usvg::MvNormal<usvg::DiagonalChol>{like_mean,  like_chol};

  auto prior_mean = blaze::DynamicVector<double>({1.0, 1.0, 1.0});
  auto prior_cov  = blaze::DynamicVector<double>({16, 16, 16});
  auto prior_chol = usvg::Cholesky<usvg::DiagonalChol>();
  REQUIRE_NOTHROW( prior_chol = usvg::cholesky_nothrow(prior_cov).value() );
  auto prior_dist = usvg::MvNormal<usvg::DiagonalChol>{prior_mean, prior_chol};

  auto loglike = [&](blaze::DynamicVector<double> const& x){
    return usvg::invquad(like_dist.cov_chol, x - like_mean) / -2;
  };

  auto x0      = prior_dist.sample(prng); 
  auto samples = elliptical_slice(prng, n_samples, n_burn, x0, loglike, prior_dist);

  auto like_inv  = 1 / like_cov;
  auto prior_inv = 1 / prior_cov;
  auto post_cov  = blaze::evaluate(1 / (like_inv + prior_inv));
  auto post_mean = blaze::evaluate(post_cov*(like_inv*like_mean + prior_inv*prior_mean));

  size_t i = 0;
  auto row = blaze::row(samples, i);
  auto cdf = [&](double x_in){
    return normal_cdf((x_in -  post_mean[i]) / sqrt(post_cov[i]));
  };
  REQUIRE( !kolmogorov_smirnoff_test(0.001, cdf, row.begin(), row.end()) );

  ++i;
  row = blaze::row(samples, i);
  REQUIRE( !kolmogorov_smirnoff_test(0.001, cdf, row.begin(), row.end()) );

  ++i;
  row = blaze::row(samples, i);
  REQUIRE( !kolmogorov_smirnoff_test(0.001, cdf, row.begin(), row.end()) );
}

