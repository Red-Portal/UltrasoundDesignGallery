
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

#include "../math/blaze.hpp"
#include "../math/cholesky.hpp"
#include "../math/linearalgebra.hpp"
#include "../math/lu.hpp"
#include "../system/debug.hpp"

#include <optional>
#include <cmath>
#include <memory>

namespace usdg
{
  template <typename CholType,
	    typename Loglike>
  inline double
  joint_likelihood(usdg::Cholesky<CholType> const& K,
		   Loglike loglike,
		   blaze::DynamicVector<double> const& f)
  {
    return loglike(f) + usdg::invquad(K, f)/-2 + usdg::logdet(K)/-2;
  }

  template <typename CholType,
	    typename LoglikeGradHess,
	    typename Loglike>
  inline std::optional<
    std::tuple<blaze::DynamicVector<double>,
	       usdg::LU,
	       blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>>>
  laplace_approximation(
    usdg::Cholesky<CholType> const& K,
    size_t n_dims,
    LoglikeGradHess loglike_grad_neghess,
    Loglike loglike,
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
   * Note: ( K^{-1} + W )^{-1} = ( K^{-1} ( I + K W )^{-1} )^{-1}
   *                           =  ( I - K W )^{-1} K
   *                           =  B^{-1} K
   */
  {
    if(log)
    {
      log->info("Starting Laplace approximation: {}", usdg::file_name(__FILE__));
      log->info("{}   {}", "iter", "||f - f*||");
    }
    auto f        = blaze::DynamicVector<double>(n_dims, 0.0);
    auto f_next   = blaze::DynamicVector<double>(f.size());
    double g_norm = std::numeric_limits<double>::max();
    double psi    = joint_likelihood(K, loglike, f);
    size_t it     = 0;
    auto Blu      = LU();
    auto W        = blaze::DynamicMatrix<double>();
    auto I        = blaze::IdentityMatrix<double>(n_dims);
    for (it = 0; it < max_iter; ++it)
    {
      auto [gradT, W_] = loglike_grad_neghess(f);
      W = std::move(W_);

      auto alpha   = usdg::solve(K, f);
      auto grad    = blaze::evaluate(gradT - alpha);
      auto KW	   = K.A*W;
      auto B	   = I + KW;
      Blu	   = usdg::lu(B);
      auto Kb	   = K.A*grad;
      auto BinvKb  = usdg::solve(Blu, Kb);
      auto p       = BinvKb;

      double stepsize = 2.0;
      double c        = 1e-2;
      double psi_next = std::numeric_limits<double>::lowest();
      double thres    = c*blaze::dot(grad, p);

      do
      {
	if(stepsize > 1e-2)
	{
	  stepsize /= 2.0;
	}
	else
	{
	  stepsize /= 1e+2;
	}

       	f_next    = f + stepsize*p; 
	psi_next  = joint_likelihood(K, loglike, f_next);
      } while(psi_next - psi < stepsize*thres);

      auto f_norm = blaze::norm(f - f_next);
      g_norm      = blaze::norm(grad);

      if(log)
      {
	log->info("{:>4}   {:g}", it,  f_norm);
      }

      if(f_norm < 1e-2 && g_norm < 1e-2)
      {
	break;
      }

      f   = f_next;
      psi = psi_next;
    }
    //std::cout << std::endl;

    if(log)
    {
      if(it == max_iter || g_norm > 1e-1)
      {
	log->warn("Laplace approximation didn't converge within {} steps.", max_iter);
      }
      else
      {
	log->info("Laplace approximation converged.");
      }
    }
    return std::tuple<decltype(f), decltype(Blu), decltype(W)>{
      std::move(f), std::move(Blu), std::move(W)};
  }
}

#endif
