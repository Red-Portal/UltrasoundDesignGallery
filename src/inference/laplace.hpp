
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

#include "../../test/finitediff.hpp"

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>

#include <optional>
#include <cmath>
#include <memory>
//#include <iostream>

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
	    typename LoglikeGradHess>
  inline blaze::DynamicVector<double>
  marginal_likelihood_gradient(usdg::Cholesky<CholType> const& K,
			       LoglikeGradHess loglike_grad_neghess,
			       blaze::DynamicVector<double> const& f)
  {
    auto [gradT, W] = loglike_grad_neghess(f);
    auto alpha      = usdg::solve(K, f);
    auto grad       = gradT - alpha;
    return grad;
  }

  template <typename CholType,
	    typename LoglikeGradHess,
	    typename Loglike>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>>
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
    auto f      = blaze::DynamicVector<double>(n_dims, 0.0);
    auto f_next = blaze::DynamicVector<double>(f.size());
    double psi  = joint_likelihood(K, loglike, f);
    size_t it   = 0;
    auto W      = blaze::DynamicMatrix<double>();
    auto I      = blaze::IdentityMatrix<double>(n_dims);
    for (it = 0; it < max_iter; ++it)
    {
      auto [gradT, W_] = loglike_grad_neghess(f);
      W = std::move(W_);

      auto alpha   = usdg::solve(K, f);
      auto grad    = blaze::evaluate(gradT - alpha);
      auto KW	   = K.A*W;
      auto B	   = I + KW;
      auto Blu	   = usdg::lu(B);
      auto Kb	   = K.A*grad;
      auto BinvKb  = usdg::solve(Blu, Kb);
      auto p       = BinvKb;

      double stepsize = 2.0;
      double c        = 1e-2;
      double psi_next = std::numeric_limits<double>::lowest();
      double graddotp = blaze::dot(grad, p);
      double thres    = 0.0;
      if(graddotp < 2e-4)
      {
	thres = c*(graddotp - graddotp/4);
      }
      else
      {
	thres = c*(graddotp - 1e-4);
      }

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

	//std::cout << stepsize << " " << psi_next << " " << psi << " " << stepsize*thres << std::endl;
      } while(psi_next - psi < stepsize*thres);
      //std::cout << '\n';

      auto f_norm = blaze::norm(f - f_next);
      auto g_norm = blaze::norm(grad);
      if(log)
      {
	log->info("{:>4}   {:g}", it,  f_norm);
      }

      if(f_norm < 1e-2 || g_norm < 1e-2)
      {
	break;
      }

      f   = f_next;
      psi = psi_next;
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
