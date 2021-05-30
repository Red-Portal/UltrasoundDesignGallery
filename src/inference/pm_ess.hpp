
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

#ifndef __US_GALLERY_PM_ESS_HPP__
#define __US_GALLERY_PM_ESS_HPP__

#include "../gp/gp_prior.hpp"
#include "../gp/kernel.hpp"
#include "../math/blaze.hpp"
#include "../math/linearalgebra.hpp"
#include "../math/mvnormal.hpp"
#include "../system/debug.hpp"
#include "ess.hpp"
#include "laplace.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <optional>
#include <type_traits>
#include <vector>

namespace usdg
{
  inline double
  logsumexp(blaze::DynamicVector<double> const& vec)
  {
    double logmin  = *std::min_element(vec.begin(), vec.end());
    size_t n_len   = vec.size();
    double res     = 0.0;
    for (size_t i = 0; i < n_len; ++i)
    {
      res += exp(vec[i] - logmin);
    }
    return log(res) + logmin;
  }

  // template <typename Loglike,
  // 	    typename CholType>
  // inline double
  // pm_likelihood(Loglike loglike,
  // 		blaze::DynamicVector<double> const& u,
  // 		usdg::Cholesky<usdg::DenseChol> const& gram_chol,
  // 		size_t n_f_dims,
  // 		size_t n_is,
  // 		usdg::MvNormal<CholType> const& dist_q_f)
  // {
  //   for (size_t i = 0; i < n_is; ++i)
  //   {
  //     auto f   = usdg::unwhiten(dist_q_f, blaze::subvector(u, i*n_f_dims, n_f_dims));
  //     auto p_f = usdg::gp_loglike(f, gram_chol);
  //     buf[i]   = loglike(f) + p_f - dist_q_f.logpdf(f);
  //   }
  //   return usdg::logsumexp(buf) - log(static_cast<double>(n_is));
  // }

  template <typename Loglike,
	    typename CholType>
  inline double
  pm_likelihood(Loglike loglike,
		blaze::DynamicVector<double> const& u,
		usdg::Cholesky<usdg::DenseChol> const& gram_chol,
		size_t n_f_dims,
		size_t n_is,
		blaze::DynamicVector<double>& buf,
		usdg::MvNormal<CholType> const& dist_q_f)
  {
    for (size_t i = 0; i < n_is; ++i)
    {
      auto f   = usdg::unwhiten(dist_q_f, blaze::subvector(u, i*n_f_dims, n_f_dims));
      auto p_f = usdg::gp_loglike(f, gram_chol);
      buf[i]   = loglike(f) + p_f - dist_q_f.logpdf(f);
    }
    return usdg::logsumexp(buf) - log(static_cast<double>(n_is));
  }

  template <typename Rng,
	    typename Loglike,
	    typename CholType>
  inline std::tuple<blaze::DynamicVector<double>, double, double>
  update_u(Rng& prng,
	   Loglike loglike,
	   blaze::DynamicVector<double> const& u,
	   double pm_prev,
	   size_t n_f_dims,
	   size_t n_is,
	   usdg::Cholesky<usdg::DenseChol> const& gram_chol,
	   usdg::MvNormal<usdg::DiagonalChol> const& u_prior,
	   usdg::MvNormal<CholType> const& dist_q_f)
  {
    auto buf    = blaze::DynamicVector<double>(n_is);
    auto target = [&](blaze::DynamicVector<double> const& u_in)->double{
      return pm_likelihood(loglike, u_in, gram_chol, n_f_dims, n_is, buf, dist_q_f);
    };
    auto [u_next, pm_next, n_props] = ess_transition(
      prng, target, u, pm_prev, u_prior);
    return {std::move(u_next), pm_next, 1.0/static_cast<double>(n_props+1)};
  }

  template <typename Rng,
	    typename Loglike,
	    typename MakeGramFunc,
	    typename GradNegHessFunc,
	    typename CholType>
  inline std::tuple<blaze::DynamicVector<double>,
		    double,
		    usdg::MvNormal<usdg::DenseChol>,
		    usdg::Cholesky<usdg::DenseChol>,
		    double>
  update_theta(Rng& prng,
	       Loglike loglike,
	       GradNegHessFunc loglike_grad_neghess,
	       MakeGramFunc  make_gram_matrix,
	       blaze::DynamicVector<double> const& theta,
	       blaze::DynamicVector<double> const& u,
	       double pm_prev,
	       size_t n_f_dims,
	       size_t n_is,
	       usdg::MvNormal<CholType> const& theta_prior,
	       bool find_feasible_init = false,
	       spdlog::logger* logger = nullptr)
  {
    size_t laplace_max_iter = 10;
    auto buf                = blaze::DynamicVector<double>(n_is);

    auto identity  = blaze::IdentityMatrix<double>(n_f_dims);
    auto dist_q_f  = usdg::MvNormal<usdg::DenseChol>();
    auto gram_chol = usdg::Cholesky<usdg::DenseChol>();
    auto target = [&](blaze::DynamicVector<double> const& theta_in)->double
    {
      /* Note: (A + B)^{-1} = L (I + L^T B L)^{-1} L^T where A^{-1} = L L^T */
      auto gram          = make_gram_matrix(theta_in);
      auto gram_chol_opt = usdg::cholesky_nothrow(gram);
      if(!gram_chol_opt)
      {
	return std::numeric_limits<double>::lowest();
      }
      gram_chol = std::move(gram_chol_opt.value());
      gram_chol_opt.reset();

      auto laplace_res = laplace_approximation(
	gram_chol,
	gram.rows(),
	loglike_grad_neghess,
	loglike,
	laplace_max_iter,
	nullptr);
      if(!laplace_res)
      {
	return std::numeric_limits<double>::lowest();
      }
      auto [f_mode, W] = laplace_res.value();

      auto IpUBL          = blaze::evaluate(identity + (blaze::trans(gram_chol.L)*W*gram_chol.L));
      auto IpUBL_chol_opt = usdg::cholesky_nothrow(IpUBL);
      if(!IpUBL_chol_opt)
      {
	return std::numeric_limits<double>::lowest();
      }
      auto& IpUBL_chol = IpUBL_chol_opt.value();

      auto laplace_half     = blaze::solve(IpUBL_chol.L, blaze::trans(gram_chol.L));
      auto laplace_cov      = blaze::trans(laplace_half) * laplace_half;
      auto laplace_chol_opt = usdg::cholesky_nothrow(laplace_cov);
      if(!laplace_chol_opt)
      {
	return std::numeric_limits<double>::lowest();
      }
      auto& laplace_chol = laplace_chol_opt.value();
      dist_q_f = usdg::MvNormal<usdg::DenseChol>{f_mode, std::move(laplace_chol)};
      laplace_chol_opt.reset();

      return pm_likelihood(loglike, u, gram_chol, n_f_dims, n_is, buf, dist_q_f);
    };

    auto theta_init = theta;
    if(find_feasible_init)
    {
      while(target(theta_init) == std::numeric_limits<double>::lowest())
      {
	if(logger)
	{
	  logger->warn("Initial hyperparameter wasn't feasible.");
	}
	theta_init = theta_prior.sample(prng);
      }
    }
    auto [theta_next, pm_next, n_props] = ess_transition(
      prng, target, theta_init, pm_prev, theta_prior);

    return {std::move(theta_next),
      pm_next,
      std::move(dist_q_f),
      std::move(gram_chol),
      1.0/static_cast<double>(n_props+1)};
  }

  template <typename Rng,
	    typename LoglikeFunc,
	    typename GradNegHessFunc,
	    typename MakeGramFunc,
	    typename CholType>
  inline std::tuple<blaze::DynamicMatrix<double>,
		    blaze::DynamicMatrix<double>,
		    std::vector<usdg::Cholesky<usdg::DenseChol>>>
  pm_ess(Rng& prng,
	 LoglikeFunc loglike,
	 GradNegHessFunc loglike_grad_neghess,
	 MakeGramFunc make_gram_matrix,
	 blaze::DynamicVector<double> const& theta_init,
	 usdg::MvNormal<CholType> const& theta_prior,
	 size_t n_dims,
	 size_t n_samples,
	 size_t n_burn,
	 size_t n_thin,
	 spdlog::logger* logger = nullptr)
  {
    size_t n_is = 16;

    auto ones     = blaze::DynamicVector<double>(n_dims*n_is, 1.0);
    auto u_prior  = MvNormal<usdg::DiagonalChol>{
      blaze::zero<double>(n_dims*n_is),
      usdg::Cholesky<DiagonalChol>{ones, ones}};
    auto u = u_prior.sample(prng);

    auto n_iters       = n_thin*n_samples;
    auto theta_samples = blaze::DynamicMatrix<double>(theta_init.size(), n_samples);
    auto f_samples     = blaze::DynamicMatrix<double>(n_dims, n_samples);
    auto gram_samples  = std::vector<usdg::Cholesky<usdg::DenseChol>>(n_samples);

    if(logger)
    {
      logger->info("Starting pseudo-marginal MCMC: {}", usdg::file_name(__FILE__));
      logger->info("{:>4}  {:>6}  {:>10}  {:>15}", "iter", "update", "acceptance", "pseudo-marginal");
    }

    auto [theta, pm, dist_q_f, gram_chol, n_props] = update_theta(
      prng,
      loglike,
      loglike_grad_neghess,
      make_gram_matrix,
      theta_init,
      u,
      std::numeric_limits<double>::lowest(),
      n_dims,
      n_is,
      theta_prior,
      true,
      logger);

    double u_accept_sum     = 0;
    double theta_accept_sum = 0;
    for (size_t i = 0; i < n_burn + n_iters; ++i)
    {
      auto [u_, pm_, u_acc] = update_u(
	prng,
	loglike,
	u,
	pm,
	n_dims,
	n_is,
	gram_chol,
	u_prior,
	dist_q_f);
      u  = u_;
      pm = pm_;

      if(logger)
      {
	u_accept_sum += u_acc;
	logger->info("{:>4}  {:>6}        {:.2f}  {:g}", i+1, 'u',
		     u_accept_sum/static_cast<double>(i+1), pm);
      }

      auto [theta_, pm__, dist_q_f_, gram_chol_, theta_acc] = update_theta(
	prng,
	loglike,
	loglike_grad_neghess,
	make_gram_matrix,
	theta,
	u,
	pm,
	n_dims,
	n_is,
	theta_prior,
	false,
	logger);

      theta     = theta_;
      pm        = pm__;
      dist_q_f  = dist_q_f_;
      gram_chol = gram_chol_;

      if(logger)
      {
	theta_accept_sum += theta_acc;
	logger->info("{:>4}  {:>6}        {:.2f}  {:g}", i+1, "θ",
		     theta_accept_sum/static_cast<double>(i+1), pm);
      }

      if(i >= n_burn && (i-n_burn) % n_thin == 0)
      {
	size_t idx_sample = (i - n_burn) / n_thin;
	blaze::column(theta_samples, idx_sample) = theta;
	blaze::column(f_samples,     idx_sample) = dist_q_f.mean;
	gram_samples[idx_sample]                 = gram_chol;
      }
    }
    return {std::move(theta_samples),
      std::move(f_samples),
      std::move(gram_samples)};
  }
}

#endif
