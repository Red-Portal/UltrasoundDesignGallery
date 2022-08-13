
/*
 * Copyright (C) 2021-2022 Kyurae Kim
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


#include "catch.hpp"
#define BLAZE_USE_DEBUG_MODE 1

#include "../src/gp/sample_beta.hpp"
#include "../src/math/prng.hpp"

#include <iostream>

TEST_CASE("beta", "[linear algebra]")
{
  auto key  = GENERATE(range(0u, 8u));
  auto prng = usdg::Random123(key);
  double alpha = 0.5;
  double lb    = 0.0;
  double ub    = 1.0;
  size_t iter  = 50;
  size_t n_samples = 1024;
  size_t n_dims    = 4;

  std::cout << blaze::trans(usdg::sample_beta(prng, alpha, lb, ub, iter, n_samples, n_dims)) << std::endl;
}
