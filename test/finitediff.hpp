
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


#ifndef __US_GALLERY_FINITEDIFF_HPP__
#define __US_GALLERY_FINITEDIFF_HPP__

#include <blaze/math/DynamicVector.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/SymmetricMatrix.h>

#include <limits>
#include <cmath>
#include <iostream>

template <typename Func, typename VecType>
inline blaze::DynamicVector<double>
finitediff_gradient(Func f, VecType const& x,
		    double eps = std::numeric_limits<double>::epsilon()*1e+2)
{
  size_t n_dims = x.size();
  auto g        = blaze::DynamicVector<double>(n_dims);
  auto delta    = blaze::DynamicVector<double>(x);
  for (size_t i = 0; i < n_dims; ++i)
  {
    double h   = sqrt(eps)*(1 + abs(x[i]));

    delta[i]   = x[i] + h;
    double fwd = f(delta);

    delta[i]   = x[i] - h;
    double bwd = f(delta);

    g[i]     = (fwd - bwd) / (2*h);
    delta[i] = x[i];
  }
  return g;
}

template <typename Func, typename VecType>
inline blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>
finitediff_hessian(Func f, VecType const& x,
		   double eps = std::numeric_limits<double>::epsilon()*1e+2)
/*
 * Abramowitz and Stegun 1972, p. 884.
 * 2n+4n2/2 function calls are needed
 */
{
  
  size_t n_dims = x.size();
  auto H        = blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>(n_dims);
  auto delta    = blaze::DynamicVector<double>(x);
  for (size_t i = 0; i < n_dims; ++i)
  {
    for (size_t j = i+1; j < n_dims; ++j)
    {
      double hi      = cbrt(eps)*(1 + abs(x[i]));
      double hj      = cbrt(eps)*(1 + abs(x[j]));

      delta[i]       = x[i] + hi;
      delta[j]       = x[j] + hj;
      double fwd_fwd = f(delta);

      delta[i]       = x[i] + hi;
      delta[j]       = x[j] - hj;
      double fwd_bwd = f(delta);

      delta[i]       = x[i] - hi;
      delta[j]       = x[j] + hj;
      double bwd_fwd = f(delta);

      delta[i]       = x[i] - hi;
      delta[j]       = x[j] - hj;
      double bwd_bwd = f(delta);
      
      H(i, j) = (fwd_fwd - fwd_bwd - bwd_fwd + bwd_bwd) / (4*hi*hj);

      delta[i] = x[i];
      delta[j] = x[j];
    }
  }

  double fx = f(x);
  for (size_t i = 0; i < n_dims; ++i)
  {
    double h    = cbrt(eps)*(1 + abs(x[i]));
    delta[i]    = x[i] + 2*h;
    double fwd1 = f(delta);

    delta[i]    = x[i] + h;
    double fwd2 = f(delta);

    delta[i]    = x[i] - h;
    double bwd1 = f(delta);

    delta[i]    = x[i] - 2*h;
    double bwd2 = f(delta);

    H(i,i)   = (-fwd1 + 16*fwd2 - 30*fx + 16*bwd1 - bwd2) / (12*h*h);

    delta[i] = x[i]; 
  }
  return H;
}

#endif
