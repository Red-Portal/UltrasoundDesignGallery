
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

#ifndef __US_GALLERY_IMH_HPP__
#define __US_GALLERY_IMH_HPP__

#include <blaze/math/DynamicVector.h>

#include <algorithm>
#include <random>
#include <stats.hpp>

namespace infer
{
  template <typename Rng, typename Func>
  inline blaze::DynamicVector<double>
  imh(Rng rng, Func f, double lb, double ub,
      size_t n_samples, size_t n_burn, size_t n_thin)
  {
    auto samples = blaze::DynamicVector<double>(n_samples / n_thin);
    double x     = stats::runif(lb, ub, rng);
    double q_cur = stats::dunif(x, lb, ub);
    double p_cur = f(x);

    for (size_t i = 0; i < n_samples + n_burn; ++i)
    {
      double x_prop = stats::runif(lb, ub, rng);
      double q_prop = stats::dunif(x_prop, lb, ub);
      double p_prop = f(x_prop);
      double alpha  = std::min(p_prop / p_cur * q_cur / q_prop, 1.0);
      double u      = stats::runif(0, 1, rng);

      if(u < alpha)
      {
	x     = x_prop;
	q_cur = q_prop;
	p_cur = p_prop;
      }
      if(i >= n_burn && (i - n_burn) % n_thin == 0)
	samples[(i - n_burn) / n_thin] = x;
    }
    return samples;
  }
}

#endif
