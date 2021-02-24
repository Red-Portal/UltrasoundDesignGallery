
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

#ifndef __US_GALLERY_GP_PRIOR_HPP__
#define __US_GALLERY_GP_PRIOR_HPP__

#include "kernel.hpp"
#include "../misc/cholesky.hpp"
#include "../misc/lu.hpp"
#include "../misc/linearalgebra.hpp"
#include "../misc/mvnormal.hpp"

#include <blaze/math/DynamicVector.h>
#include <blaze/math/DynamicMatrix.h>

#include <numbers>

namespace usdg
{
  template <typename KernelFunc>
  struct LatentGaussianProcess
  {
    usdg::Cholesky<usdg::DenseChol> cov_chol;
    blaze::DynamicVector<double>    alpha;
    KernelFunc                      kernel;
    
    inline std::tuple<double, double>
    predict(blaze::DynamicMatrix<double> const& data,
	    blaze::DynamicVector<double> const& x) const;
  };

  template <typename KernelFunc>
  inline std::tuple<double, double>
  LatentGaussianProcess<KernelFunc>::
  predict(blaze::DynamicMatrix<double> const& data,
	  blaze::DynamicVector<double> const& x) const
  /* 
   * Predictive mean and variance.
   * mean = k(x) K^{-1} f
   * var  = k(x, x) - k(x)^T (K + W^{-1})^{-1} k(x)
   */
  {
    size_t n_data = data.rows();
    auto k_star   = blaze::DynamicVector<double>(n_data);
    for (size_t i = 0; i < n_data; ++i)
    {
      k_star[i] = this->kernel(blaze::column(data, i), x);
    }
    auto k_self   = this->kernel(x, x);
    auto mean     = blaze::dot(k_star, alpha);
    double gp_var = usdg::invquad(this->cov_chol, k_star);
    auto var      = k_self - gp_var;
    return {mean, var};
  }

  inline std::tuple<double, blaze::DynamicVector<double>>
  gp_loglike(blaze::DynamicVector<double> const& f,
	     usdg::Cholesky<usdg::DenseChol> const& cov_chol)
  {
    size_t n_dims     = f.size();
    double normalizer = log(2*std::numbers::pi);
    double D          = static_cast<double>(n_dims);
    auto alpha        = usdg::solve(cov_chol, f);
    double like = (blaze::dot(alpha, f)
		   + usdg::logdet(cov_chol)
		   + D*normalizer)/-2;
    return {like, std::move(alpha)};
  }

  template <typename Rng>
  inline blaze::DynamicVector<double>
  sample_gp_prior(Rng& prng,
		  usdg::Matern52 const& kernel,
		  blaze::DynamicMatrix<double> const& points)
  {
    auto K      = usdg::compute_gram_matrix(kernel, points);
    auto K_chol = usdg::Cholesky<usdg::DenseChol>();
    REQUIRE_NOTHROW( K_chol = usdg::cholesky_nothrow(K).value() );
  
    auto Z = usdg::rmvnormal(prng, K.rows());
    auto y = K_chol.L * Z;
    return y;
  }
}

#endif
