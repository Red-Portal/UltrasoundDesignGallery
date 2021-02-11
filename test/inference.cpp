
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
#include "../src/misc/seed.hpp"

#include <iostream>

TEST_CASE("Sampling from unit Gaussian", "[imh]")
{
  auto prng = generate_seed(1);
  auto p    = [](double x){
    return stats::dnorm(x, 0.0, 1.0);
  };
  size_t n_samples = 4096;
  auto samples = infer::imh(prng, p, -3, 3, n_samples, 128, 2);

  double sample_mean_margin = 1.0 / sqrt(static_cast<double>(n_samples)) * 10;
  REQUIRE( blaze::mean(samples)   == Approx(0.0).margin(sample_mean_margin) );
  REQUIRE( blaze::stddev(samples) == Approx(1.0).margin(0.1) );
}
