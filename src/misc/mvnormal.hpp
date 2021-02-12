
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
#include <random>

namespace usvg
{
  template<typename CholType>
  inline double
  dmvnormal(blaze::DynamicVector<double> const& x,
	    blaze::DynamicVector<double> const& mu,
	    usvg::Cholesky<CholType> const& cov_chol,
	    bool logdensity = false)
  {
    size_t n_dims = x.size();
    double constexpr normalizer = log(2 * std::numbers::pi);
    double D    = static_cast<double>(n_dims);
    double logp = (usvg::logdet(cov_chol)
		   + usvg::invquad(cov_chol, mu - x)
		   + D*normalizer)/-2;
    if(logdensity)
      return logp;
    else
      return exp(logp);
  }

  template <typename Rng>
  inline blaze::DynamicVector<double>
  rmvnormal(Rng& prng, size_t n_dims) noexcept
  {
    auto res  = blaze::DynamicVector<double>(n_dims); 
    auto dist = std::normal_distribution<double>(0.0, 1.0);
    for (size_t i = 0; i < n_dims; ++i)
    {
      res[i] = dist(prng);
    }
    return res;
  }

  template <typename Rng, typename CholType>
  inline blaze::DynamicVector<double>
  rmvnormal(Rng& prng,
	    blaze::DynamicVector<double> const& mu,
	    usvg::Cholesky<CholType> const& cov_chol)
  {
    size_t n_dims = mu.size();
    auto z        = rmvnormal(prng, n_dims);
    return cov_chol.L*z + mu;
  }
}

#endif