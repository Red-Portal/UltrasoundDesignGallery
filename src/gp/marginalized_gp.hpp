
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

#ifndef __US_GALLERY_MARGINALIZEDGP_HPP__
#define __US_GALLERY_MARGINALIZEDGP_HPP__

#include "../gp/data.hpp"
#include "../gp/gp_prior.hpp"
#include "../gp/likelihood.hpp"
#include "../inference/pm_ess.hpp"
#include "../math/blaze.hpp"

#include <vector>

namespace usdg
{
  template <typename Kernel>
  class MarginalizedGP
  {
    std::vector<Kernel>                           _k_samples;
    blaze::DynamicMatrix<double>                  _alpha_samples;
    std::vector<usdg::Cholesky<usdg::DenseChol>>  _K_samples;

  public:
    inline
    MarginalizedGP(blaze::DynamicMatrix<double> const& theta_samples,
		 blaze::DynamicMatrix<double> const& f_samples,
		 std::vector<usdg::Cholesky<usdg::DenseChol>> const& K_samples) ;

    inline
    MarginalizedGP(blaze::DynamicMatrix<double> const& theta_samples,
		   blaze::DynamicMatrix<double> const& f_samples,
		   std::vector<usdg::Cholesky<usdg::DenseChol>>&& K_samples) ;

    inline std::pair<double, double>
    predict(blaze::DynamicMatrix<double> const& data,
	    blaze::DynamicVector<double> const& x) const;
  };

  template <typename Kernel>
  inline
  MarginalizedGP<Kernel>::
  MarginalizedGP(blaze::DynamicMatrix<double> const& theta_samples,
		 blaze::DynamicMatrix<double> const& f_samples,
		 std::vector<usdg::Cholesky<usdg::DenseChol>>&& K_samples)
    : _k_samples(theta_samples.columns()),
      _alpha_samples(f_samples.rows(), f_samples.columns()),
      _K_samples(theta_samples.columns())
  {
    size_t n_samples = theta_samples.columns();
    for (size_t i = 0; i < n_samples; ++i)
    {
      _k_samples[i]  = Kernel(blaze::column(theta_samples, i));
      blaze::column(_alpha_samples, i) =
	usdg::solve(K_samples[i], blaze::column(f_samples,i));
      _K_samples[i] = std::move(K_samples[i]);
    }
  }

  template <typename Kernel>
  inline
  MarginalizedGP<Kernel>::
  MarginalizedGP(blaze::DynamicMatrix<double> const& theta_samples,
		 blaze::DynamicMatrix<double> const& f_samples,
		 std::vector<usdg::Cholesky<usdg::DenseChol>> const& K_samples)
    : _k_samples(theta_samples.columns()),
      _alpha_samples(f_samples.rows(), f_samples.columns()),
      _K_samples(theta_samples.columns())
  {
    size_t n_samples = theta_samples.columns();
    for (size_t i = 0; i < n_samples; ++i)
    {
      _k_samples[i]  = Kernel(blaze::column(theta_samples, i));
      blaze::column(_alpha_samples, i) =
	usdg::solve(K_samples[i], blaze::column(f_samples,i));
      _K_samples[i] = K_samples[i];
    }
  }

  template <typename KernelFunc>
  inline std::pair<double, double>
  MarginalizedGP<KernelFunc>::
  predict(blaze::DynamicMatrix<double> const& data,
	  blaze::DynamicVector<double> const& x) const
  /* 
   * Predictive mean and variance.
   * mean = k(x) K^{-1} f
   * var  = k(x, x) - k(x)^T (K + W^{-1})^{-1} k(x)
   */
  {
    size_t n_samples = _k_samples.size();
    double mean      = 0.0;
    double var       = 0.0;
    for (size_t i = 0; i < n_samples; ++i)
    {
      auto [mean_i, var_i] = usdg::predict(this->_k_samples[i],
					   data,
					   this->_K_samples[i],
					   blaze::column(this->_alpha_samples, i),
					   x);
      mean += mean_i;
      var  += var_i;
    }
    double n = static_cast<double>(n_samples);
    return { mean/n, var/n };
  }
}

#endif
