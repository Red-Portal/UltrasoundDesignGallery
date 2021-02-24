
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

#include "../src/misc/prng.hpp"
#include "../src/misc/uniform.hpp"
#include "../src/misc/mvnormal.hpp"
#include "statistical_test.hpp"

#include <random>

TEST_CASE("Null-hypothesis is true", "[statistical test]")
{
  auto prng    = usdg::Random123();
  auto dist    = std::normal_distribution<double>(0.0, 1.0);
  auto samples = blaze::DynamicVector<double>(512);
  std::generate(samples.begin(), samples.end(),
		[&]{ return dist(prng); });
  REQUIRE( !kolmogorov_smirnoff_test(0.01, usdg::normal_cdf,
				     samples.begin(), samples.end()) );
}

TEST_CASE("Null-hypothesis is false", "[statistical test]")
{
  auto prng    = usdg::Random123();
  auto samples = blaze::DynamicVector<double>(512);
  std::generate(samples.begin(), samples.end(),
		[&]{ return usdg::runiform(prng, -3, 3); });
  REQUIRE( kolmogorov_smirnoff_test(0.01, usdg::normal_cdf,
				    samples.begin(), samples.end()) );
}
