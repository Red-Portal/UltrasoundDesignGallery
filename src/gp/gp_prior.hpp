
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
#include "../math/blaze.hpp"
#include "../math/cholesky.hpp"
#include "../math/lu.hpp"
#include "../math/linearalgebra.hpp"
#include "../math/mvnormal.hpp"

#include <numbers>

namespace usdg
{
  template <typename KernelFunc>
  struct GP
  {
    usdg::Cholesky<usdg::DenseChol> cov_chol;
    blaze::DynamicVector<double>    alpha;
    KernelFunc                      kernel;
    
    template <typename MatType, typename VecType>
    inline decltype(auto)
    predict(MatType const& data, VecType const& x) const;
  };

  template <typename Kernel,
	    typename MatType,
	    typename AlphaVecType,
	    typename XVecType>
  inline std::pair<double, double>
  predict(Kernel const& kernel,
	  MatType const& data,
	  usdg::Cholesky<usdg::DenseChol> const& cov_chol,
	  AlphaVecType const& alpha,
	  XVecType const& x)
  {
    size_t n_data = data.columns();
    auto k_star   = blaze::DynamicVector<double>(n_data);
    for (size_t i = 0; i < n_data; ++i)
    {
      k_star[i] = kernel(blaze::column(data, i), x);
    }
    auto k_self   = kernel(x, x);
    auto mean     = blaze::dot(k_star, alpha);
    double gp_var = usdg::invquad(cov_chol, k_star);
    auto var      = k_self - gp_var;
    return {mean, var};
  }

  template <typename Kernel,
	    typename MatType,
	    typename AlphaVecType>
  inline std::pair<blaze::DynamicVector<double>,
		   blaze::DynamicVector<double>>
  predict(Kernel const& kernel,
	  MatType const& data,
	  usdg::Cholesky<usdg::DenseChol> const& cov_chol,
	  AlphaVecType const& alpha,
	  blaze::DynamicMatrix<double> const& X)
  {
    size_t n_data  = data.columns();
    size_t n_input = X.columns();
    auto k_stars   = blaze::DynamicMatrix<double>(n_data, n_input);
    for (size_t j = 0; j < n_input; ++j) {
      for (size_t i = 0; i < n_data; ++i) {
	k_stars(i, j) = kernel(blaze::column(data, i), blaze::column(X, j));
      }
    }
    auto k_selves  = blaze::DynamicVector<double>(n_input);
    for (size_t i = 0; i < n_input; ++i)
    {
      k_selves[i] = kernel(blaze::column(X, i), blaze::column(X, i));
    }
    auto means   = blaze::trans(blaze::trans(alpha)*k_stars);
    auto gp_vars = usdg::invquad_batch(cov_chol, k_stars);
    auto vars    = k_selves - gp_vars;
    return {means, vars};
  }

  template <typename KernelFunc>
  template <typename MatType, typename VecType>
  inline decltype(auto)
  GP<KernelFunc>::
  predict(MatType const& data, VecType const& x) const
  {
    return usdg::predict(this->kernel, data, this->cov_chol, this->alpha, x);
  }

  inline std::tuple<double, blaze::DynamicVector<double>>
  gp_loglike_alpha(blaze::DynamicVector<double> const& f,
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

  inline double
  gp_loglike(blaze::DynamicVector<double> const& f,
	     usdg::Cholesky<usdg::DenseChol> const& cov_chol)
  {
    size_t n_dims     = f.size();
    double normalizer = log(2*std::numbers::pi);
    double D          = static_cast<double>(n_dims);
    double like = (usdg::invquad(cov_chol, f)
		   + usdg::logdet(cov_chol)
		   + D*normalizer)/-2;
    return like;
  }

  template <typename Rng>
  inline blaze::DynamicVector<double>
  sample_gp_prior(Rng& prng,
		  usdg::Matern52ARD const& kernel,
		  blaze::DynamicMatrix<double> const& points)
  {
    auto K      = usdg::compute_gram_matrix(kernel, points);
    auto K_chol = usdg::Cholesky<usdg::DenseChol>();
    K_chol      = usdg::cholesky_nothrow(K).value();
  
    auto Z = usdg::rmvnormal(prng, K.rows());
    auto y = K_chol.L * Z;
    return y;
  }

  // template <typename KernelFunc,
  // 	    typename MatType,
  // 	    typename VecType>
  // inline blaze::DynamicVector<double>
  // gradient_predict(usdg::GP<KernelFunc> const& gp,
  // 		   MatType const& data,
  // 		   VecType const& dx)
  // {

  //   size_t n_data  = data.columns();
  //   size_t n_dims  = data.rows();
  //   auto pred_grad = blaze::DynamicVector<double>(n_dims, 0.0);
  //   auto 
  //   auto sigma2   = gp.kernel.sigma * gp.kernel.sigma;
  //   for (size_t i = 0; i < n_data; ++i)
  //   {
  //     auto y       = blaze::column(data, i);
  //     auto kstardx = derivative(gp.kernel, sigma2, dx, y);
  //     grad        += kstardx*gp.alpha[i];
  //   }
  //   return grad;
  // }

  template <typename KernelFunc,
	    typename MatType,
	    typename VecType>
  inline std::pair<double, blaze::DynamicVector<double>>
  gradient_mean(usdg::GP<KernelFunc> const& gp,
		MatType const& data,
		VecType const& dx)
  {
    size_t n_data = data.columns();
    size_t n_dims = data.rows();
    auto& kernel  = gp.kernel;
    auto grad     = blaze::DynamicVector<double>(n_dims, 0.0);
    auto sigma2   = gp.kernel.sigma * gp.kernel.sigma;

    auto k_star    = blaze::DynamicVector<double>(n_data);
    for (size_t i  = 0; i < n_data; ++i)
    {
      k_star[i] = kernel(blaze::column(data, i), dx);
    }

    auto mean = blaze::dot(k_star, gp.alpha);
    for (size_t i = 0; i < n_data; ++i)
    {
      auto y       = blaze::column(data, i);
      auto kstardx = usdg::gradient(gp.kernel, sigma2, dx, y);
      grad        += kstardx*gp.alpha[i];
    }
    return { mean, std::move(grad) };
  }

  template <typename KernelFunc,
	    typename MatType,
	    typename VecType>
  inline std::tuple<double,
		    double,
		    blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
  gradient_mean_var(usdg::GP<KernelFunc> const& gp,
		    MatType const& data,
		    VecType const& dx)
  {
    size_t n_data  = data.columns();
    size_t n_dims  = data.rows();
    auto mean_grad = blaze::DynamicVector<double>(n_dims, 0.0);
    auto var_grad  = blaze::DynamicVector<double>(n_dims, 0.0);
    auto& kernel   = gp.kernel;
    auto sigma2    = kernel.sigma*kernel.sigma;
    auto k_star    = blaze::DynamicVector<double>(n_data);
    for (size_t i  = 0; i < n_data; ++i)
    {
      k_star[i]    = kernel(blaze::column(data, i), dx);
    }
    auto Kinvkstar = usdg::solve(gp.cov_chol, k_star);

    for (size_t i  = 0; i < n_data; ++i)
    {
      auto y       = blaze::column(data, i);
      auto kstardx = usdg::gradient(gp.kernel, sigma2, dx, y);
      mean_grad   += kstardx*gp.alpha[i];
      var_grad    += kstardx*Kinvkstar[i];
    }

    auto k_self   = kernel(dx, dx);
    auto mean     = blaze::dot(k_star, gp.alpha);
    double gp_var = usdg::invquad(gp.cov_chol, k_star);
    auto var      = k_self - gp_var;
    var_grad     *= -2;
    return { mean, var, std::move(mean_grad), std::move(var_grad) };
  }
}

#endif
