
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
#include "../src/math/prng.hpp"
#include "../src/math/mvnormal.hpp"
#include "../src/math/cholesky.hpp"

#include "finitediff.hpp"
#include "utils.hpp"

TEST_CASE("gaussian process prediction gradient", "[kernel]")
{
  auto key        = GENERATE(range(0u, 8u));
  auto prng       = usdg::Random123(key);
  size_t n_dims   = 8;
  size_t n_points = 32;

  auto data_mat  = generate_mvsamples(prng, n_dims, n_points);
  auto norm_dist = std::normal_distribution<double>(0, 1);
  auto sigma     = exp(norm_dist(prng));
  auto scale     = exp(norm_dist(prng) + 1);
  auto kernel    = usdg::Matern52Iso{sigma, scale};
  auto gram      = usdg::compute_gram_matrix(kernel, data_mat);
  auto z         = usdg::rmvnormal(prng, n_points);
  auto dx        = usdg::rmvnormal(prng, n_dims);
  auto chol      = usdg::cholesky_nothrow(gram).value();

  auto gp         = usdg::LatentGaussianProcess<decltype(kernel)>{
    std::move(chol), z, kernel};
  auto grad_truth = finitediff_gradient(
    [&data_mat, &gp](blaze::DynamicVector<double> const& x)
    {
      auto [mean, var] = gp.predict(data_mat, x);
      return mean;
    }, dx);
  auto grad = usdg::gradient_mean(gp, data_mat, dx);

  REQUIRE( blaze::norm(grad_truth - grad) < 1e-4 );
}
