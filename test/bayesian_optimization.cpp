
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

#include "../src/bo/find_bounds.hpp"
#include "../src/bo/acquisition.hpp"
#include "../src/gp/gp_prior.hpp"
#include "../src/math/cholesky.hpp"
#include "../src/math/mvnormal.hpp"
#include "../src/math/prng.hpp"

#include "utils.hpp"

TEST_CASE("Thompson sampling gradient", "[bo]")
{
  auto key        = GENERATE(range(0u, 8u));
  auto prng       = usdg::Random123(key);
  size_t n_dims   = 8;
  size_t n_points = 32;

  auto data_mat  = generate_mvsamples(prng, n_dims, n_points);
  auto norm_dist = std::normal_distribution<double>(0, 1);
  auto sigma     = exp(norm_dist(prng));
  auto scale     = exp(norm_dist(prng) + 2);
  auto kernel    = usdg::SquaredExpIso{sigma, scale};
  auto gram      = usdg::compute_gram_matrix(kernel, data_mat);
  auto z         = usdg::rmvnormal(prng, n_points);
  auto dxi       = usdg::rmvnormal(prng, n_dims);
  auto chol      = usdg::cholesky_nothrow(gram).value();
  auto x         = usdg::rmvnormal(prng, n_dims);
  size_t n_beta  = 32;
  auto gp        = usdg::GP<decltype(kernel)>{std::move(chol), z, kernel};

  auto func = [&data_mat, &x, &gp, n_beta](blaze::DynamicVector<double> const& xi_in)
    {
      auto [lb, ub]   = usdg::pbo_find_bounds(x, xi_in);
      auto beta_delta = (ub - lb)/(static_cast<double>(n_beta) - 1);
      auto y_avg      = 0.0;
      for (size_t i = 0; i < n_beta; ++i)
      {
	auto beta        = lb + beta_delta*static_cast<double>(i);
	auto [mean, var] = gp.predict(data_mat, x + beta*xi_in);
	y_avg += mean;
      }
      return y_avg/static_cast<double>(n_beta);
    };

  auto grad_truth  = finitediff_gradient(func, dxi);
  auto [val, grad] = usdg::thompson_xi_gradient(gp, data_mat, n_beta, x, dxi, true);

  REQUIRE( func(dxi) == Approx(val) );
  REQUIRE( blaze::norm(grad_truth - grad) < 1e-6 );
}
