
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

#ifndef __US_GALLERY_LIKELIHOOD_HPP__
#define __US_GALLERY_LIKELIHOOD_HPP__

#include "data.hpp"
#include "../misc/blaze.hpp"
#include "../misc/mvnormal.hpp"
#include "../misc/quadrature.hpp"

#include <cmath>
#include <limits>
#include <numbers>

namespace usdg
{
  inline std::tuple<double, double>
  pgp_find_bounds(blaze::DynamicVector<double> const& x,
		  blaze::DynamicVector<double> const& xi)
  {
    double lb     = std::numeric_limits<double>::lowest();
    double ub     = std::numeric_limits<double>::max();
    size_t n_dims = x.size();

    for (size_t i = 0; i < n_dims; ++i)
    {
      double alpha = (1 - x[i]) / xi[i];
      if(alpha > 0)
      {
	ub = std::min(alpha, ub);
      }
      else
      {
	lb = std::max(alpha, lb);
      }

      alpha = -x[i] / xi[i];
      if(alpha > 0)
      {
	ub = std::min(alpha, ub);
      }
      else
      {
	lb = std::max(alpha, lb);
      }
    }
    return {lb, ub};
  }

  inline double
  dnormal2(double x) noexcept
  {
    return 1/(2*sqrt(std::numbers::pi)) * exp(x*x / -4);
  }

  inline double
  pgp_loglike(blaze::DynamicMatrix<double> const& delta)
  {
    double m = static_cast<double>(delta.columns());
    double sqrt2  = sqrt(2);
    double sqrtpi = sqrt(std::numbers::pi);
    auto pdfcdfconv = blaze::map(
      delta, 
      [sqrtpi, sqrt2](double x)->double{
	return usdg::gauss_hermite([x, sqrt2](double x_itgr){
	  return usdg::normal_cdf(x - x_itgr*sqrt2) ;
	}) / sqrtpi;
      });
    return blaze::sum(pdfcdfconv) / -m;
  }

  inline blaze::DynamicMatrix<double>
  pgp_delta(blaze::DynamicVector<double> const& f,
	    usdg::Dataset const& data,
	    double sigma)
  {
    size_t n_data   = data.num_data();
    size_t n_pseudo = data.num_pseudo();
    auto delta      = blaze::DynamicMatrix<double>(n_data, n_pseudo);

    for (size_t data_idx = 0; data_idx < n_data; ++data_idx)
    {
      for (size_t pseudo_idx = 0; pseudo_idx < n_pseudo; ++pseudo_idx)
      {
	size_t alpha_idx = data.alpha_index(data_idx);
	size_t beta_idx  = data.beta_index(data_idx, pseudo_idx);
	delta(data_idx, pseudo_idx) = (f[beta_idx] - f[alpha_idx]) / sigma;
      }
    }
    return delta;
  }

  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>>
  pgp_loglike_gradneghess(blaze::DynamicMatrix<double> const& delta,
			  usdg::Dataset const& data,
			  double sigma)
  {
    size_t n_data   = data.num_data();
    size_t n_pseudo = data.num_pseudo();
    size_t n_f      = n_data*(n_pseudo + 1);
    auto phi    = blaze::evaluate(blaze::map(delta, [](double x){ return usdg::dnormal2(x); }));
    auto grad   = blaze::DynamicVector<double>(n_f);
    auto hess   = blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>(
      blaze::zero<double>(n_f, n_f));
    auto sigma2 = sigma*sigma;
    double m    = static_cast<double>(n_pseudo);

    for (size_t data_idx = 0; data_idx < n_data; ++data_idx)
    {
      size_t alpha_idx = data.alpha_index(data_idx);
      grad[alpha_idx]  = blaze::sum(blaze::row(phi, data_idx)) / sigma / m;
      for (size_t pseudo_idx = 0; pseudo_idx < n_pseudo; ++pseudo_idx)
      {
	size_t beta_idx = data.beta_index(data_idx, pseudo_idx);
	grad[beta_idx]  = -phi(pseudo_idx, data_idx) / sigma / m;
      }
    }

    for (size_t data_idx = 0; data_idx < n_data; ++data_idx)
    {
      /* alpha_i, alpha_i */
      size_t alpha_idx = data.alpha_index(data_idx);
      auto phi_col     = blaze::row(phi,   data_idx);
      auto delta_col   = blaze::row(delta, data_idx);
      hess(alpha_idx, alpha_idx) = blaze::dot(phi_col, delta_col) / 2 / sigma2 / m;

      /* alpha_i, beta_j */
      for (size_t pseudo_idx = 0; pseudo_idx < n_pseudo; ++pseudo_idx)
      {
	size_t beta_idx = data.beta_index(data_idx, pseudo_idx);
	auto delta_ij   = delta_col[pseudo_idx];
	auto phi_ij     = phi_col[pseudo_idx];

	hess(alpha_idx, beta_idx) = -delta_ij*phi_ij / 2 / sigma2 / m;
	hess(beta_idx, beta_idx)  =  delta_ij*phi_ij / 2 / sigma2 / m;
      }
    }

    return {std::move(grad), -1*hess};
  }
}

#endif
