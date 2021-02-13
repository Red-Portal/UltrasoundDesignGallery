
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

#include <cmath>
#include <numbers>
#include <random>

namespace usvg
{
  template <typename CholType>
  struct MvNormal
  {
    blaze::DynamicVector<double> mean;
    usvg::Cholesky<CholType> cov_chol;

    inline double pdf(blaze::DynamicVector<double>    const& x) const;

    inline double logpdf(blaze::DynamicVector<double> const& x) const;

    template <typename Rng>
    inline blaze::DynamicVector<double> sample(Rng& prng) const;
  };

  template <typename CholType>
  inline double
  dmvnormal(blaze::DynamicVector<double> const& x,
	    blaze::DynamicVector<double> const& mean,
	    usvg::Cholesky<CholType> const& cov_chol,
	    bool logdensity = false)
  {
    size_t n_dims     = x.size();
    double normalizer = log(2*std::numbers::pi);
    double D          = static_cast<double>(n_dims);
    double logp       = (usvg::logdet(cov_chol)
			 + usvg::invquad(cov_chol, x - mean)
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
	    blaze::DynamicVector<double> const& mean,
	    usvg::Cholesky<CholType> const& cov_chol)
  {
    size_t n_dims = mean.size();
    auto z        = usvg::rmvnormal(prng, n_dims);
    return cov_chol.L*z + mean;
  }

  template <typename CholType>
  inline double
  MvNormal<CholType>::
  pdf(blaze::DynamicVector<double>    const& x) const
  {
    return usvg::dmvnormal(x, this->mean, this->cov_chol);
  }

  template <typename CholType>
  inline double
  MvNormal<CholType>::
  logpdf(blaze::DynamicVector<double> const& x) const
  {
    return usvg::dmvnormal(x, this->mean, this->cov_chol, true);
  }

  template <typename CholType>
  template <typename Rng>
  inline blaze::DynamicVector<double>
  MvNormal<CholType>::
  sample(Rng& prng) const
  {
    return usvg::rmvnormal(prng, this->mean, this->cov_chol);
  }
}

#endif
