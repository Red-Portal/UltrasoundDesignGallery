
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

#include "../src/gp/kernel.hpp"

double const catch_eps = 1e-8;

TEST_CASE("Matern 5/2 kernel value", "[kernel]")
{
  /* 
   * Compared against Julia KernelFunctions.jl result

   using KernelFunctions
   ℓ = [0.3468645418042131, 0.4089279965964634]; σ² = 0.6
   k = KernelFunctions.Matern52Kernel()
   t = KernelFunctions.ARDTransform(1 ./ℓ)
   k = σ²*KernelFunctions.transform(k, t)
   x = [1.124618098544101, -1.8477787735615157]; y = [1.0597907259031794, 0.20131396456561368]
   k(x, y)

   */
  auto linescales = blaze::DynamicVector<double>({0.3468645418042131, 0.4089279965964634});
  auto sigma      = sqrt(0.6);
  auto kernel     = usdg::Matern52{sigma, linescales};

  auto x = blaze::DynamicVector<double>({1.124618098544101, -1.8477787735615157});
  auto y = blaze::DynamicVector<double>({1.0597907259031794, 0.20131396456561368});

  REQUIRE(kernel(x, y) == Approx(0.0004385141317002246));
}

TEST_CASE("Gram matrix computation", "[kernel]")
{
  /* 
   * Compared against Julia KernelFunctions.jl result

   using KernelFunctions
   ℓ = [0.3468645418042131, 0.4089279965964634]; σ² = 0.6
   k = KernelFunctions.Matern52Kernel()
   t = KernelFunctions.ARDTransform(1 ./ℓ)
   k = σ²*KernelFunctions.transform(k, t)
   x = [1.124618098544101, -1.8477787735615157]; y = [1.0597907259031794, 0.20131396456561368]

   data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
   kernelmatrix(k, data, obsdim=1)
   */

  auto linescales = blaze::DynamicVector<double>(
    {0.3468645418042131, 0.4089279965964634});
  auto sigma      = sqrt(0.6);
  auto kernel     = usdg::Matern52{sigma, linescales};
  auto datamatrix = blaze::DynamicMatrix<double>(
    {{1.0, 3.0, 5.0},
     {2.0, 4.0, 6.0}});
  auto truth = blaze::DynamicMatrix<double>(
    {{0.6,         3.08681e-6, 5.15601e-13},
     {3.08681e-6,  0.6,        3.08681e-6},
     {5.15601e-13, 3.08681e-6, 0.6}});

  auto gram = usdg::compute_gram_matrix(kernel, blaze::trans(datamatrix));
  REQUIRE( blaze::norm(gram - truth) < catch_eps );
}
