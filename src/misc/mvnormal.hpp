
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

#include "cholesky.hpp"
#include "linearalgebra.hpp"
#include "blaze.hpp"

#include <cmath>
#include <numbers>
#include <random>
#include <iostream>

namespace usdg
{
  template <typename CovType>
  struct MvNormal
  {
    blaze::DynamicVector<double> mean;
    usdg::Cholesky<CovType> cov_chol;

    inline double pdf(blaze::DynamicVector<double>    const& x) const;

    inline double logpdf(blaze::DynamicVector<double> const& x) const;

    template <typename Rng>
    inline blaze::DynamicVector<double> sample(Rng& prng) const;
  };

  struct UnitNormal {};

  template <>
  struct MvNormal<UnitNormal>
  {
    size_t n_dims;

    inline MvNormal(size_t n_dims_);

    inline double pdf(blaze::DynamicVector<double>    const& x) const;

    inline double logpdf(blaze::DynamicVector<double> const& x) const;

    template <typename Rng>
    inline blaze::DynamicVector<double> sample(Rng& prng) const;
  };

  inline
  MvNormal<UnitNormal>::
  MvNormal(size_t n_dims_)
    : n_dims(n_dims_)
  { }

  struct LaplaceNormal {};

  template <>
  struct MvNormal<LaplaceNormal>
  {
    blaze::DynamicVector<double> mean;

    blaze::LowerMatrix<blaze::DynamicMatrix<double>> cov_L;
     
    blaze::LowerMatrix<blaze::DynamicMatrix<double>> IpUBL_L;

    inline double logpdf(blaze::DynamicVector<double> const& x) const;
  };

  template <typename CovType>
  inline double
  dmvnormal(blaze::DynamicVector<double> const& x,
	    blaze::DynamicVector<double> const& mean,
	    usdg::Cholesky<CovType> const& cov_chol,
	    bool logdensity = false)
  {
    size_t n_dims     = x.size();
    double normalizer = log(2*std::numbers::pi);
    double D          = static_cast<double>(n_dims);
    double logp       = (usdg::logdet(cov_chol)
			 + usdg::invquad(cov_chol, x - mean)
			 + D*normalizer)/-2;
    if(logdensity)
      return logp;
    else
      return exp(logp);
  }

  template <typename Rng>
  inline blaze::DynamicVector<double>
  rmvnormal(Rng& prng, size_t n_dims)
  {
    auto res  = blaze::DynamicVector<double>(n_dims); 
    auto dist = std::normal_distribution<double>(0.0, 1.0);
    for (size_t i = 0; i < n_dims; ++i)
    {
      res[i] = dist(prng);
    }
    return res;
  }

  inline double
  dmvnormal(blaze::DynamicVector<double> const& x,
	    bool logdensity = false) noexcept
  {
    size_t n_dims     = x.size();
    double normalizer = log(2*std::numbers::pi);
    double D          = static_cast<double>(n_dims);
    double logp       = (blaze::dot(x, x) + D*normalizer)/-2;
    if(logdensity)
      return logp;
    else
      return exp(logp);
  }

  template <typename CovType>
  inline blaze::DynamicVector<double>
  unwhiten(blaze::DynamicVector<double> const& mean,
	   CovType const& L,
	   blaze::DynamicVector<double> const& z)
  {
    return L*z + mean; 
  }

  template <typename CovType>
  inline blaze::DynamicVector<double>
  unwhiten(usdg::MvNormal<CovType> const& dist,
	   blaze::DynamicVector<double> const& z)
  {
    return unwhiten(dist.mean, dist.cov_chol.L, z); 
  }

  inline blaze::DynamicVector<double>
  unwhiten(usdg::MvNormal<usdg::LaplaceNormal> const& dist,
	   blaze::DynamicVector<double> const& z)
  {
    auto cov_U = blaze::trans(dist.cov_L);
    auto Ux    = cov_U * z;
    auto x     = blaze::solve(dist.IpUBL_L, Ux) + dist.mean;
    return x;
  }

  template <typename Rng, typename CovType>
  inline blaze::DynamicVector<double>
  rmvnormal(Rng& prng,
	    blaze::DynamicVector<double> const& mean,
	    usdg::Cholesky<CovType> const& cov_chol)
  {
    size_t n_dims = mean.size();
    auto z        = usdg::rmvnormal(prng, n_dims);
    return unwhiten(mean, cov_chol.L, z);
  }

  template <typename CovType>
  inline double
  MvNormal<CovType>::
  pdf(blaze::DynamicVector<double>    const& x) const
  {
    return usdg::dmvnormal(x, this->mean, this->cov_chol);
  }

  template <typename CovType>
  inline double
  MvNormal<CovType>::
  logpdf(blaze::DynamicVector<double> const& x) const
  {
    return usdg::dmvnormal(x, this->mean, this->cov_chol, true);
  }

  template <typename CovType>
  template <typename Rng>
  inline blaze::DynamicVector<double>
  MvNormal<CovType>::
  sample(Rng& prng) const
  {
    return usdg::rmvnormal(prng, this->mean, this->cov_chol);
  }

  inline double
  MvNormal<UnitNormal>::
  pdf(blaze::DynamicVector<double>    const& x) const
  {
    return usdg::dmvnormal(x, false);
  }

  inline double
  MvNormal<UnitNormal>::
  logpdf(blaze::DynamicVector<double> const& x) const
  {
    return usdg::dmvnormal(x, true);
  }

  inline double
  MvNormal<LaplaceNormal>::
  logpdf(blaze::DynamicVector<double> const& x) const
  {
    auto x_delta = x - this->mean;
    auto IpUBL_U = blaze::trans(this->IpUBL_L);
    auto z       = IpUBL_U * blaze::solve(this->cov_L, x_delta);

    double logdetcov  = 2*(usdg::logtrace(this->cov_L)
			   - usdg::logtrace(this->IpUBL_L));
    size_t n_dims     = x.size();
    double normalizer = log(2*std::numbers::pi);
    double D          = static_cast<double>(n_dims);
    double logp       = (logdetcov + blaze::dot(z, z) + D*normalizer)/-2;
    return logp;
  }

  template <typename Rng>
  inline blaze::DynamicVector<double>
  MvNormal<UnitNormal>::
  sample(Rng& prng) const
  {
    return usdg::rmvnormal(prng, this->n_dims);
  }

  inline double
  normal_cdf(double x)
  {
    return std::erfc(-x/std::sqrt(2))/2;
  }
}

#endif
