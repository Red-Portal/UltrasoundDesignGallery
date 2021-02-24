
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

#include "../src/gp/likelihood.hpp"
#include "../src/gp/data.hpp"
#include "../src/misc/prng.hpp"
#include "finitediff.hpp"

// template <typename Rng, typename Func, typename Gen>
// usvg::Dataset
// generate_dataset(Rng& prng, Func func, Gen gen, size_t n_points)
// {
//   auto dist = std::normal_distribution<double>(0, 1);
//   for (size_t i = 0; i < n_points; ++i)
//   {
//     auto x  = blaze::DynamicVector<double>(gen);
//     auto xi = blaze::DynamicVector<double>(gen);
    
//   }
// }

TEST_CASE("preferential gaussian process likelihood derivatives", "[likelihood]")
{
  size_t n_dims   = 2;
  size_t n_pseudo = 4;
  size_t n_data   = 16;

  auto key     = GENERATE(range(0u, 8u));
  auto prng    = usdg::Random123(key);
  auto dist    = std::normal_distribution<double>(0, 1);
  auto vecgen  = blaze::generate(n_dims,   [&prng, &dist](size_t)->double { return dist(prng); });
  auto betagen = blaze::generate(n_pseudo, [&prng, &dist](size_t)->double { return dist(prng); });
  auto data    = usdg::Dataset(n_dims, n_pseudo);
  for (size_t i = 0; i < n_data; ++i)
  {
    auto x     = blaze::DynamicVector<double>(vecgen);
    auto xi    = blaze::DynamicVector<double>(vecgen);
    auto alpha = dist(prng);
    auto beta  = blaze::DynamicVector<double>(betagen);
    auto dp    = usdg::Datapoint{alpha, std::move(beta), std::move(xi), std::move(x)};
    data.push_back(dp);
  }

  size_t n_latent = data.num_data()*(1 + data.num_pseudo());
  auto f          = blaze::DynamicVector<double>(
    blaze::generate(n_latent, [&prng, &dist](size_t)->double { return dist(prng); }));

  auto loglike = [&](blaze::DynamicVector<double> const& f_in)->double{
    auto delta = usdg::pgp_delta(f_in, data, 1.0);
    return usdg::pgp_loglike(delta);
  };

  auto delta  = usdg::pgp_delta(f, data, 1.0);
  auto [g, H] = usdg::pgp_loglike_gradhess(delta, data, 1.0);

  auto g_truth = finitediff_gradient(loglike, f);
  auto H_truth = finitediff_hessian(loglike,  f);

  REQUIRE(blaze::norm(g - g_truth) < 1e-4);

  /* diagonal */
  REQUIRE(blaze::norm(blaze::diagonal(H) - blaze::diagonal(H_truth)) < 1e-4);

  /* off-diagonal */
  blaze::diagonal(H)       = 0;
  blaze::diagonal(H_truth) = 0;
  REQUIRE(blaze::norm(H - H_truth) < 1e-4);
}


