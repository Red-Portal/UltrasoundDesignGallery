
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

#include "../src/inference/imh.hpp"
#include "../src/misc/mvnormal.hpp"
#include "../src/misc/prng.hpp"
#include "statistical_test.hpp"

#include <cmath>

TEST_CASE("Sampling from unit Gaussian", "[imh]")
{
  auto key  = GENERATE(range(0u, 8u));
  auto prng = usdg::Random123(key);
  auto p    = [](double x){
    return exp(-x*x/2);
  };
  size_t n_samples = 512;
  auto samples = usdg::imh(prng, p, -3, 3, n_samples, 128, 1);

  REQUIRE( !kolmogorov_smirnoff_test(0.001, usdg::normal_cdf,
				     samples.begin(), samples.end()) );
}
