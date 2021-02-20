
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
  template <typename CovType>
  struct MvNormal
  {
    blaze::DynamicVector<double> mean;
    usvg::Cholesky<CovType> cov_chol;

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
     
    blaze::LowerMatrix<blaze::DynamicMatrix<double>> IpLBL_L;

    inline double logpdf(blaze::DynamicVector<double> const& x) const;
  };

  template <typename CovType>
  inline double
  dmvnormal(blaze::DynamicVector<double> const& x,
	    blaze::DynamicVector<double> const& mean,
	    usvg::Cholesky<CovType> const& cov_chol,
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
  unwhiten(usvg::MvNormal<CovType> const& dist,
	   blaze::DynamicVector<double> const& z)
  {
    return unwhiten(dist.mean, dist.cov_chol.L, z); 
  }

  inline blaze::DynamicVector<double>
  unwhiten(usvg::MvNormal<usvg::LaplaceNormal> const& dist,
	   blaze::DynamicVector<double> const& z)
  {
    auto Lz       =  dist.cov_L * z;
    auto x        =  blaze::solve(
      blaze::declupp(
	blaze::trans(dist.IpLBL_L)), Lz) + dist.mean;
    return x;
  }

  template <typename Rng, typename CovType>
  inline blaze::DynamicVector<double>
  rmvnormal(Rng& prng,
	    blaze::DynamicVector<double> const& mean,
	    usvg::Cholesky<CovType> const& cov_chol)
  {
    size_t n_dims = mean.size();
    auto z        = usvg::rmvnormal(prng, n_dims);
    return unwhiten(mean, cov_chol.L, z);
  }

  template <typename CovType>
  inline double
  MvNormal<CovType>::
  pdf(blaze::DynamicVector<double>    const& x) const
  {
    return usvg::dmvnormal(x, this->mean, this->cov_chol);
  }

  template <typename CovType>
  inline double
  MvNormal<CovType>::
  logpdf(blaze::DynamicVector<double> const& x) const
  {
    return usvg::dmvnormal(x, this->mean, this->cov_chol, true);
  }

  template <typename CovType>
  template <typename Rng>
  inline blaze::DynamicVector<double>
  MvNormal<CovType>::
  sample(Rng& prng) const
  {
    return usvg::rmvnormal(prng, this->mean, this->cov_chol);
  }

  inline double
  MvNormal<UnitNormal>::
  pdf(blaze::DynamicVector<double>    const& x) const
  {
    return usvg::dmvnormal(x, false);
  }

  inline double
  MvNormal<UnitNormal>::
  logpdf(blaze::DynamicVector<double> const& x) const
  {
    return usvg::dmvnormal(x, true);
  }

  inline double
  MvNormal<LaplaceNormal>::
  logpdf(blaze::DynamicVector<double> const& x) const
  {
    auto x_delta = x - this->mean;
    auto z       = blaze::trans(this->IpLBL_L)*blaze::solve(this->cov_L, x_delta);

    double logdetcov  = 2*(usvg::logtrace(this->cov_L) - usvg::logtrace(this->IpLBL_L));
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
    return usvg::rmvnormal(prng, this->n_dims);
  }
}

#endif
