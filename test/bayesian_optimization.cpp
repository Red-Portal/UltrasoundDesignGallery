
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

#include "../src/bo/find_bounds.hpp"
#include "../src/bo/acquisition.hpp"
#include "../src/gp/gp_prior.hpp"
#include "../src/gp/sample_beta.hpp"
#include "../src/math/cholesky.hpp"
#include "../src/math/mvnormal.hpp"
#include "../src/math/prng.hpp"

#include "finitediff.hpp"
#include "utils.hpp"

TEST_CASE("expected improvement gradient with x", "[bo]")
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
  auto dx        = usdg::rmvnormal(prng, n_dims);
  auto chol      = usdg::cholesky_nothrow(gram).value();
  auto x         = usdg::rmvnormal(prng, n_dims);
  auto y_opt     = blaze::max(z);
  auto gp        = usdg::GP<decltype(kernel)>{std::move(chol), z, kernel};

  auto func = [&](blaze::DynamicVector<double> const& x_in)
    {
      auto [value, _] = usdg::ei_with_deidx(gp, data_mat, y_opt, x_in, false);
      return value; 
    };

  auto grad_truth  = finitediff_gradient(func, dx);
  auto [val, grad] = usdg::ei_with_deidx(gp, data_mat, y_opt, dx, true);

  REQUIRE( func(dx) == Catch::Approx(val) );
  REQUIRE( blaze::norm(grad_truth - grad) < 1e-6 );
}
