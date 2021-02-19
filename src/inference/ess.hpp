
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

#ifndef __US_GALLERY_ESS_HPP__
#define __US_GALLERY_ESS_HPP__

#include "../misc/mvnormal.hpp"
#include "../misc/uniform.hpp"

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>

#include <algorithm>
#include <iostream>
#include <cmath>
#include <random>

namespace usvg
{
  template <typename Rng,
	    typename Loglike,
	    typename CholType>
  inline std::tuple<blaze::DynamicVector<double>, double, size_t>
  ess_transition(Rng& rng,
		 Loglike loglike,
		 blaze::DynamicVector<double> const& x_prev,
		 double loglike_prev,
		 usvg::MvNormal<CholType> const& prior)
  /*
   * Elliptical slice sampler Markov-chain kernel.
   *
   * Murray, Iain, Ryan Adams, and David MacKay. 
   * "Elliptical slice sampling." 
   * AISTATS, 2010.
   *
   * Nishihara, Robert, Iain Murray, and Ryan P. Adams. 
   * "Parallel MCMC with generalized elliptical slice sampling." 
   * JMLR 15.1 (2014): 2087-2112.
   */
  {
    auto const tau = 2*std::numbers::pi;
    auto nu        = prior.sample(rng);
    auto u         = usvg::runiform(rng);
    auto logy      = loglike_prev + log(u);
    auto theta     = usvg::runiform(rng, 0, tau);
    auto theta_min = theta - tau;
    auto theta_max = theta;
    size_t n_props = 1;

    while(true)
    {
      auto costh = cos(theta);
      auto sinth = sin(theta);
      auto a     = 1 - (costh + sinth);

      auto x_prop       = costh*x_prev + sinth*nu + a*prior.mean;
      auto loglike_prop = loglike(x_prop);

      if(loglike_prop > logy)
      { /* Accept */
	return {x_prop, loglike_prop, n_props};
      }
      else
      { /* Reject. Shrink bracket */
	if(theta < 0)
	{
	  theta_min = theta;
	}
	else
	{
	  theta_max = theta;
	}
	theta = usvg::runiform(rng, theta_min, theta_max);
	++n_props;
      }
    }
  }
}

#endif
