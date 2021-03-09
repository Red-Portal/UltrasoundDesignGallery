
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

#include "../src/gp/gp_prior.hpp"
#include "../src/gp/kernel.hpp"
#include "../src/inference/ess.hpp"
#include "../src/math/blaze.hpp"
#include "../src/math/cholesky.hpp"
#include "../src/math/linearalgebra.hpp"
#include "../src/math/mvnormal.hpp"
#include "../src/math/prng.hpp"
#include "elliptical_slice.hpp"
#include "statistical_test.hpp"
#include "utils.hpp"

#include <cmath>

TEST_CASE("Identifiability check of GP hyperparameters using ESS", "[gp & ess]")
{
  auto key        = GENERATE(range(0u, 8u));
  auto prng       = usdg::Random123(key);
  size_t n_points = 128;
  size_t n_dims   = 3;

  auto prior_mean = blaze::zero<double>(n_dims+1);
  auto prior_var  = blaze::DynamicVector<double>(n_dims+1, 2.0);
  auto prior_chol = usdg::Cholesky<usdg::DiagonalChol>();
  REQUIRE_NOTHROW( prior_chol = usdg::cholesky_nothrow(prior_var).value() );
  auto prior_dist = usdg::MvNormal<usdg::DiagonalChol>{prior_mean, prior_chol};

  auto truth  = prior_dist.sample(prng);
  auto kernel = usdg::Matern52ARD{
    exp(truth[0]), blaze::exp(blaze::subvector(truth, 1, n_dims))};

  auto data_x = generate_mvsamples(prng, n_dims, n_points);
  auto data_y = usdg::sample_gp_prior(prng, kernel, data_x);
  
  auto mll = [&data_x, &data_y, n_dims, n_points](
    blaze::DynamicVector<double> const& theta)->double{
    auto _sigma      = exp(theta[0]);
    auto _linescales = exp(blaze::subvector(theta, 1u, n_dims));
    auto _kernel     = usdg::Matern52ARD{_sigma, _linescales};
    auto _K          = usdg::compute_gram_matrix(_kernel, data_x); 
    auto zero_mean   = blaze::zero<double>(n_points);
    if(auto _K_chol = usdg::cholesky_nothrow(_K))
      return usdg::dmvnormal(data_y, zero_mean, _K_chol.value(), true);
    else
      return std::numeric_limits<double>::lowest();
  };

  size_t n_samples = 512;
  size_t n_burnin  = 256;
  auto x0          = prior_dist.sample(prng); 
  auto samples     = elliptical_slice(prng, n_samples, n_burnin, x0, mll, prior_dist);

  for (size_t i = 0; i < n_dims+1; ++i)
  { /* Posterior contraction and posterior z-score */
    auto mu = blaze::mean(blaze::row(samples, i));
    auto s  = blaze::stddev(blaze::row(samples, i));
    auto z  = (mu - truth[i]) / s;
    auto c = 1 - ((s*s)/prior_var[i]);

    REQUIRE(abs(z) < 3);
    REQUIRE(c      > 0.5);
  }
}
