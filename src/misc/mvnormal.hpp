
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

#ifndef __US_GALLERY_MVNORMAL_HPP__
#define __US_GALLERY_MVNORMAL_HPP__

#include "linearalgebra.hpp"
#include "cholesky.hpp"

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/LowerMatrix.h>
#include <blaze/math/DynamicVector.h>

#include <numbers>

namespace usvg
{
  template <typename F>
  inline double
  dmvnormal(blaze::DynamicVector<F> const& x,
	    blaze::DynamicVector<F> const& mu,
	    usvg::Cholesky const& cov_chol,
	    bool logdensity = false)
  {
    size_t n_dims = x.size();
    double constexpr normalizer = log(2 * std::numbers::pi);
    double D    = static_cast<double>(n_dims);
    double logp = (usvg::logdet(cov_chol.L)
		   + usvg::invquad(cov_chol.L, mu - x)
		   + D*normalizer)/-2;
    if(logdensity)
      return logp;
    else
      return exp(logp);
  }

  template <typename F>
  inline blaze::DynamicVector<F>
  rmvnormal()
  {
  
  }
}

#endif
