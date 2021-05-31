
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

#ifndef __US_GALLERY_GPINFERENCE_HPP__
#define __US_GALLERY_GPINFERENCE_HPP__

#include "../gp/gp_prior.hpp"
#include "../gp/likelihood.hpp"
#include "../inference/laplace.hpp"

#include <nlopt.hpp>

namespace usdg
{
  inline double
  logprior(blaze::DynamicVector<double> const& theta_in)
  {
    double sigmalike_mean   = 0.0;
    double sigmalike_sd     = 1.0;

    double sigmanoise_mean  = 0.0;
    double sigmanoise_sd    = 1.0;

    return usdg::dnormal((theta_in[0] - sigmalike_mean) / sigmalike_sd, true)
      + usdg::dnormal((theta_in[1] - sigmanoise_mean) / sigmanoise_sd, true);
  }

  template <typename Rng,
	    typename MarginalLogLikeFunc>
  inline blaze::DynamicVector<double>
  map_inference(Rng& prng,
		MarginalLogLikeFunc mll,
		spdlog::logger* logger = nullptr)
  {
    size_t n_dims = 2;
    auto x_init   = std::vector<double>(n_dims, 0.0);
    auto x_buf    = blaze::DynamicVector<double>(n_dims, 0.0);
    auto objective_lambda = [&mll, &x_buf](
      std::vector<double> const& x,
      std::vector<double>&) -> double
    {
      std::copy(x.begin(), x.end(), x_buf.begin());
      return mll(x_buf);
    };

    auto objective_wrapped = std::function<
      double(std::vector<double> const&,
	     std::vector<double>&)>(objective_lambda);

    auto objective_invoke = +[](std::vector<double> const& x,
				std::vector<double>& grad,
				void* punned)
    {
      return (*reinterpret_cast<
	std::function<
	      double(std::vector<double> const&,
		     std::vector<double>&)>*>(punned))(x, grad);
    };
    auto optimizer = nlopt::opt(nlopt::LN_NELDERMEAD,
				static_cast<unsigned int>(x_init.size()));
    optimizer.set_max_objective(objective_invoke, &objective_wrapped);
    optimizer.set_xtol_rel(1e-4);
    optimizer.set_ftol_rel(1e-5);
    optimizer.set_maxeval(1024);
    nlopt::srand(static_cast<unsigned long>(prng()));

    double y_buf = mll(x_buf);
    optimizer.optimize(x_init, y_buf);
    auto x_found = blaze::DynamicVector<double>(n_dims);
    std::copy(x_init.begin(), x_init.end(), x_found.begin());

    std::cout << mll(x_buf) << " " << y_buf << std::endl;
    return x_found;
  }

  inline usdg::Cholesky<usdg::DenseChol>
  laplace_marginal_covariance(
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double>> const& W,
    usdg::Cholesky<usdg::DenseChol> const& K_chol)
  {
    size_t n_f_dims      = K_chol.L.rows();
    auto I               = blaze::IdentityMatrix<double>(n_f_dims);
    auto IpLtBL          = blaze::evaluate(I + (blaze::trans(K_chol.L)*W*K_chol.L));
    auto IpLtBL_chol_opt = usdg::cholesky_nothrow(IpLtBL);
    if(!IpLtBL_chol_opt)
    {
      throw std::runtime_error("Failed to fit Gaussian process");
    }
    return std::move(IpLtBL_chol_opt.value());
  }

  template <typename Rng>
  inline decltype(auto)
  fit_gp(Rng& prng,
	 usdg::Dataset const& data,
	 blaze::DynamicMatrix<double> const& data_mat,
	 blaze::DynamicVector<double> const& linescales,
	 spdlog::logger* logger = nullptr)
  {
    auto n_dims      = data_mat.columns();
    double sigma_buf = 0.1;
    auto I           = blaze::IdentityMatrix<double>(data_mat.columns());
    auto make_gram   = [&](blaze::DynamicVector<double> const& theta_in)
      ->blaze::DynamicMatrix<double>
      {
	auto kernel = usdg::Matern52ARD{1.0, linescales};
	auto gram   = usdg::compute_gram_matrix(kernel, data_mat);
	sigma_buf   = exp(theta_in[0]);
	return gram + (exp(theta_in[1])*I);
      };
    
    auto grad_neghess = [&](blaze::DynamicVector<double> const& f_in)
      ->std::tuple<blaze::DynamicVector<double>,
		   blaze::DynamicMatrix<double>>
      {
	auto delta = usdg::pgp_delta(f_in, data, sigma_buf);
	return usdg::pgp_loglike_gradneghess(delta, data, sigma_buf);
      };

    auto loglike = [&](blaze::DynamicVector<double> const& f_in){
      auto delta = usdg::pgp_delta(f_in, data, sigma_buf);
      return usdg::pgp_loglike(delta);
    };

    auto laplace_marginal = [&](blaze::DynamicVector<double> const& theta_in){
      auto gram          = make_gram(theta_in);
      auto gram_chol_opt = usdg::cholesky_nothrow(gram);
      if(!gram_chol_opt)
      {
	return std::numeric_limits<double>::lowest();
      }
      auto gram_chol   = std::move(gram_chol_opt.value());
      auto laplace_res = laplace_approximation(gram_chol,
					       n_dims,
					       grad_neghess,
					       loglike,
					       20,
					       nullptr);
      if(!laplace_res)
      {
	return std::numeric_limits<double>::lowest();
      }
      auto [ f_mode, Blu, _ ] = laplace_res.value();
      double t1    = loglike(f_mode);
      double t2    = usdg::invquad(gram_chol, f_mode) / -0.5;
      double t3    = abs(usdg::logabsdet(Blu)) / -0.5 ;
      double prior = logprior(theta_in);
      return t1 + t2 + t3 + prior;
    };

    auto theta = map_inference(prng, laplace_marginal);
    auto gram  = make_gram(theta);
    auto gram_chol_opt = usdg::cholesky_nothrow(gram);
    if(!gram_chol_opt)
    {
      throw std::runtime_error("Failed to fit Gaussian process");
    }
    auto gram_chol   = std::move(gram_chol_opt.value());
    auto laplace_res = laplace_approximation(gram_chol,
					     n_dims,
					     grad_neghess,
					     loglike,
					     20,
					     logger);
    if(!laplace_res)
    {
      throw std::runtime_error("Failed to fit Gaussian process");
    }
    auto [f_mode, _, W]   = laplace_res.value();
    auto laplacecov_chol  = laplace_marginal_covariance(W, gram_chol);
    auto kernel           = usdg::Matern52ARD{1.0, linescales};
    auto alpha            = usdg::solve(laplacecov_chol, f_mode);
    return usdg::GP<decltype(kernel)>{
      std::move(laplacecov_chol),
      std::move(alpha),
      std::move(kernel)};
  } 
}

#endif
