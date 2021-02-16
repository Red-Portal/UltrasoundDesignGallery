
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

#include "../misc/cholesky.hpp"
#include "../misc/lu.hpp"
#include "../misc/linearalgebra.hpp"

#include <blaze/math/DynamicVector.h>
#include <blaze/math/DynamicMatrix.h>

namespace usvg
{
  template <typename KernelFunc>
  struct LatentGaussianProcess
  {
    usvg::Cholesky<usvg::DenseChol> K;
    blaze::DynamicVector<double>    alpha;
    blaze::DynamicMatrix<double>    data;
    blaze::DynamicMatrix<double>    WK;
    usvg::LU                        IpWK; /* LU of (I + WK) */
    KernelFunc                      kernel;
    
    inline std::tuple<double, double>
    predict(blaze::DynamicVector<double> const& x) const;
  };

  template <typename KernelFunc>
  inline std::tuple<double, double>
  LatentGaussianProcess<KernelFunc>::
  predict(blaze::DynamicVector<double> const& x) const
  /* 
   * Predictive mean and variance.
   * mean = k(x) K^{-1} f
   * var  = k(x, x) - k(x)^T (K + W^{-1})^{-1} k(x)
   *
   * ( K^{-1} + W )^{-1} = K ( I - ( I + W K )^{-1} ) W K 
   * = K ( WK - (I + W K) \ WK )
   */
  {
    size_t n_data = K.A.rows();
    auto k_star   = blaze::DynamicVector<double>(n_data);
    for (size_t i = 0; i < n_data; ++i)
    {
      k_star[i] = this->kernel(blaze::column(this->data, i), x);
    }
    auto k_self = this->kernel(x, x);
    auto  mean  = blaze::dot(k_star, alpha);

    auto WKkstar     = this->WK*k_star;
    auto IpWKinvb    = usvg::solve(this->IpWK, WKkstar);
    auto WpKinvkstar = this->K.A * (WKkstar - IpWKinvb);
    auto kKpWk       = blaze::dot(WpKinvkstar, k_star);
    auto var         = k_self - kKpWk;
    return {mean, var};
  }
}

#endif
