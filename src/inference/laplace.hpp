
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

#ifndef __US_GALLERY_LAPLACE_HPP__
#define __US_GALLERY_LAPLACE_HPP__

#include "../misc/cholesky.hpp"
#include "../misc/debug.hpp"
#include "../misc/linearalgebra.hpp"
#include "../misc/lu.hpp"

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>

#include <optional>
#include <memory>

namespace usdg
{
  template <typename LoglikeGradHess>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>>
  laplace_approximation(
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double>> const& K,
    blaze::DynamicVector<double> const& f0,
    LoglikeGradHess loglike_grad_neghess,
    size_t max_iter = 20,
    spdlog::logger* log = nullptr)
  /*
   * Variant of the Newton's method based mode-locating algorithm (GPML, Algorithm 3.1)
   * Utilizes the Woodburry identity for avoiding doing two matrix decompositions 
   * per Newton iteration.
   *
   * Reduces the stepsize whenever the marginal likelhood gets stuck
   * Algortihm 3.1 utilizes the fact that W is diagonal which is not for our case.
   *
   * Note: ( K^{-1} + W )^{-1} = K ( I - ( I + W K )^{-1} W K ) 
   *                           = K ( I - B^{-1} W K ) 
   */
  {
    size_t n_dims = f0.size(); 
    auto f        = blaze::DynamicVector<double>(f0);

    if(log)
    {
      log->info("Starting Laplace approximation: {}", usdg::file_name(__FILE__));
      log->info("{}   {}", "iter", "||f - f*||");
    }

    size_t it = 0;
    auto W    = blaze::DynamicMatrix<double>();
    auto I    = blaze::IdentityMatrix<double>(n_dims);
    for (it = 0; it < max_iter; ++it)
    {
      auto [gradT, W_] = loglike_grad_neghess(f);
      W = W_;

      auto b	   = W*f + gradT;
      auto WK	   = W*K;
      auto B	   = I + WK;
      auto Blu	   = usdg::lu(B);
      auto WKb	   = WK*b;
      auto BinvWKb = usdg::solve(Blu, WKb);
      auto a	   = b - BinvWKb;
      auto f_next  = K*a;

      auto delta_f = blaze::norm(f - f_next);
      f            = f_next;

      if(log)
      {
	log->info("{:>4}   {:g}", it, delta_f);
      }

      if(delta_f < 1e-2)
      {
	break;
      }
    }

    if(log && it == max_iter)
    {
      log->warn("Laplace approximation didn't converge within {} steps.", max_iter);
    }
    else if(log)
    {
      log->info("Laplace approximation converged.");
    }

    return {std::move(f), std::move(W)};
  }
}

#endif
