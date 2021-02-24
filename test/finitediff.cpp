
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
#include "finitediff.hpp"

#include <blaze/util/Random.h>

#include <random>
#include <iostream>

TEST_CASE("gradient", "[finitediff]")
{
  auto key  = GENERATE(range(0u, 8u));
  auto prng = usvg::Random123(key);
  auto dist = std::normal_distribution<double>(0, 1);
  size_t n  = 128;

  auto A = blaze::DynamicMatrix<double>(
    blaze::generate(n, n, [&prng, &dist](size_t,size_t)->double { return dist(prng); }));
  auto x = blaze::DynamicVector<double>(
    blaze::generate(n,    [&prng, &dist](size_t)->double        { return dist(prng); }));

  auto f = [&](blaze::DynamicVector<double> const& x_in){
    return blaze::dot(x_in, A*x_in);
  };

  auto g_truth = (A + blaze::trans(A))*x;
  auto g       = finitediff_gradient(f, x);
  REQUIRE(blaze::norm(g - g_truth) < 1e-4);
}

TEST_CASE("hessian", "[finitediff]")
{
  auto key  = GENERATE(range(0u, 8u));
  auto prng = usvg::Random123(key);
  auto dist = std::normal_distribution<double>(0, 1);
  size_t n  = 128;

  auto A = blaze::DynamicMatrix<double>(
    blaze::generate(n, n, [&prng, &dist](size_t,size_t)->double { return dist(prng); }));
  auto x = blaze::DynamicVector<double>(
    blaze::generate(n,    [&prng, &dist](size_t)->double        { return dist(prng); }));

  auto f = [&](blaze::DynamicVector<double> const& x_in){
    return blaze::dot(x_in, A*x_in);
  };

  auto g_truth = blaze::evaluate(A + blaze::trans(A));
  auto g       = finitediff_hessian(f, x);

  /* diagonal */
  REQUIRE(blaze::norm(blaze::diagonal(g) - blaze::diagonal(g_truth)) < 1e-4);
  
  /* off-diagonal */
  blaze::diagonal(g)       = 0;
  blaze::diagonal(g_truth) = 0;
  REQUIRE(blaze::norm(g - g_truth) < 1e-4);
}
