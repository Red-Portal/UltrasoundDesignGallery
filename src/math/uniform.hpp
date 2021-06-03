
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

#ifndef __US_GALLERY_UNIFORM_HPP__
#define __US_GALLERY_UNIFORM_HPP__

#include <cmath>
#include <random>

#include "blaze.hpp"

namespace usdg
{
  inline double
  duniform(double,
	   double a,
	   double b,
	   bool logdensity = false)
  {
    if(logdensity)
    {
      return -log(b - a);
    }
    else
    {
      return 1 / (b - a);
    }
  }

  template <typename Rng>
  inline double
  runiform(Rng& prng)
  {
    auto dist = std::uniform_real_distribution<double>(0, 1);
    return dist(prng);
  }

  template <typename Rng>
  inline double
  runiform(Rng& prng,
	   double a,
	   double b)
  {
    auto dist = std::uniform_real_distribution<double>(a, b);
    return dist(prng);
  }

  template <typename Rng>
  inline blaze::DynamicVector<double>
  rmvuniform(Rng& prng,
	     size_t n_dims,
	     double a,
	     double b)
  {
    auto res  = blaze::DynamicVector<double>(n_dims); 
    auto dist = std::uniform_real_distribution<double>(a, b);
    for (size_t i = 0; i < n_dims; ++i)
    {
      res[i] = dist(prng);
    }
    return res;
  }
}

#endif
