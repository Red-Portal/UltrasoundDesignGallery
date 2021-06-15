
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

#ifndef __US_GALLERY_SGD_HPP__
#define __US_GALLERY_SGD_HPP__

#include <algorithm>
#include <cmath>
#include <iostream>

#include "../math/blaze.hpp"
#include "../math/uniform.hpp"

namespace usdg
{
  template <typename ObjFunc,
	    typename Proj,
	    typename Rng>
  inline blaze::DynamicVector<double>
  spsa_maximize(Rng& prng,
		ObjFunc obj,
		Proj proj,
		double noise_stddev,
		double stepsize,
		blaze::DynamicVector<double> const& x_init,
		size_t n_iters)
  {
    /* 
     * Simultaneous Perturbation
     * "An Overview of the Simultaneous Perturbation Method for Efficient Optimization"
     * Spall, James C., 1998
     * 
     * Parameters suggested by 
     * "Implementation of the Simultaneous Perturbation 
     *  Algorithm for Stochastic Optimization"
     * by Spall, James C., 1998, IEEE Tran. Aerospace Electronic Syst.
     */
    double c     = noise_stddev;
    double alpha = 0.602;
    double gamma = 0.101;
    double A     = static_cast<double>(n_iters)/10;
    double a     = stepsize * pow(A + 1, alpha);
    double p     = 0.5;

    auto x          = blaze::DynamicVector<double>(x_init);
    size_t n_dims   = x.size();
    auto delta      = blaze::DynamicVector<double>(n_dims);
    auto delta_dist = std::bernoulli_distribution(p);
    for (size_t t = 1; t <= n_iters; ++t)
    {
      double ak  = a / pow(static_cast<double>(t) + A, alpha);
      double ck  = c / pow(static_cast<double>(t), gamma);
      
      for (size_t i = 0; i < n_dims; ++i)
      {
	if(delta_dist(prng))
	  delta[i] = 1;
	else
	  delta[i] = -1;
      }

      auto x_plus   = x + ck*delta;
      auto x_minus  = x - ck*delta;
      double yplus  = obj(x_plus);
      double yminus = obj(x_minus);
      auto ghat     = (yplus - yminus) / (2*ck*delta);

      if(t == 1)
      {
	a /= blaze::max(blaze::abs(ghat));
      }

      x += ak*ghat;
      x  = proj(x);
    }
    return x;
  }
}

#endif
