
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
#include "../misc/blaze.hpp"
#include "../misc/debug.hpp"
#include "../misc/linearalgebra.hpp"
#include "../misc/mvnormal.hpp"
#include "ess.hpp"
#include "laplace.hpp"

#include <numbers>
#include <vector>
#include <cmath>
#include <iostream>

namespace usdg
{
  template <typename Rng,
	    typename Loglike,
	    typename CholType>
  inline std::tuple<blaze::DynamicVector<double>, double, size_t>
  update_u(Rng& prng,
	   Loglike loglike,
	   blaze::DynamicVector<double> const& u,
	   double pm_prev,
	   usdg::Cholesky<usdg::DenseChol> const& gram_chol,
	   usdg::MvNormal<usdg::DiagonalChol> const& u_prior,
	   usdg::MvNormal<CholType> const& dist_q_f)
  {
    auto target = [&](blaze::DynamicVector<double> const& u_in)->double{
      auto f            = usdg::unwhiten(dist_q_f, u_in);
      auto [p_f, alpha] = usdg::gp_loglike(f, gram_chol);
      double pm_f       = loglike(f) + p_f - dist_q_f.logpdf(f);
      return pm_f;
    };
    auto [u_next, pm_next, n_props] = ess_transition(
      prng, target, u, pm_prev, u_prior);
    return {std::move(u_next), pm_next, n_props};
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
		    size_t>
  update_theta(Rng& prng,
	       Loglike loglike,
	       GradNegHessFunc loglike_grad_neghess,
	       MakeGramFunc  make_gram_matrix,
	       blaze::DynamicVector<double> const& theta,
	       blaze::DynamicVector<double> const& u,
	       double pm_prev,
	       usdg::MvNormal<CholType> const& theta_prior,
	       spdlog::logger* logger = nullptr)
  {
    size_t laplace_max_iter = 20;
    size_t n_dims           = u.size();

    auto identity  = blaze::IdentityMatrix<double>(n_dims);
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
	logger);
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

      // auto smallest_diag = blaze::min(blaze::diagonal(gram_chol.L))
      // 	/ blaze::max(blaze::diagonal(IpUBL_chol.L));
      // if(smallest_diag < 1e-5)
      // {
      // 	return std::numeric_limits<double>::lowest();
      // }
      
      auto f            = usdg::unwhiten(dist_q_f, u);
      auto [p_f, alpha] = usdg::gp_loglike(f, gram_chol);
      auto pm_f         = p_f + loglike(f) - dist_q_f.logpdf(f);
      return pm_f;
    };
    auto [theta_next, pm_next, n_props] = ess_transition(
      prng, target, theta, pm_prev, theta_prior);

    return {std::move(theta_next),
      pm_next,
      std::move(dist_q_f),
      std::move(gram_chol),
      n_props};
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
	 spdlog::logger* logger = nullptr)
  {
    auto ones     = blaze::DynamicVector<double>(n_dims, 1.0);
    auto u_prior  = MvNormal<usdg::DiagonalChol>{
      blaze::zero<double>(n_dims),
      usdg::Cholesky<DiagonalChol>{ones, ones}};
    auto u        = MvNormal<usdg::UnitNormal>(n_dims).sample(prng);

    auto theta_samples = blaze::DynamicMatrix<double>(theta_init.size(), n_samples);
    auto f_samples     = blaze::DynamicMatrix<double>(u.size(), n_samples);
    auto gram_samples  = std::vector<usdg::Cholesky<usdg::DenseChol>>(n_samples);

    auto [theta, pm, dist_q_f, gram_chol, n_props] = update_theta(
      prng,
      loglike,
      loglike_grad_neghess,
      make_gram_matrix,
      theta_init,
      u,
      std::numeric_limits<double>::lowest(),
      theta_prior,
      logger);

    if(logger)
    {
      logger->info("Starting pseudo-marginal MCMC: {}", usdg::file_name(__FILE__));
      logger->info("{:>4}  {:>6}  {:>10}  {:>15}", "iter", "update", "acceptance", "pseudo-marginal");
    }

    size_t n_total_props_u     = 0;
    size_t n_total_props_theta = 0;
    for (size_t i = 0; i < n_burn + n_samples; ++i)
    {
      auto [u_, pm_, n_props_u] = update_u(
	prng,
	loglike,
	u,
	pm,
	gram_chol,
	u_prior,
	dist_q_f);
      u  = u_;
      pm = pm_;

      if(logger)
      {
	n_total_props_u += n_props_u;
	double acc_u     = static_cast<double>(i+1)/static_cast<double>(n_total_props_u);
	logger->info("{:>4}  {:>6}        {:.2f}  {:g}", i+1, 'u', acc_u, pm);
      }

      auto [theta_, pm__, dist_q_f_, gram_chol_, n_props_theta] = update_theta(
	prng,
	loglike,
	loglike_grad_neghess,
	make_gram_matrix,
	theta,
	u,
	pm,
	theta_prior);
      //logger);

      theta     = theta_;
      pm        = pm__;
      dist_q_f  = dist_q_f_;
      gram_chol = gram_chol_;

      if(logger)
      {
	n_total_props_theta += n_props_theta;
	double acc_theta     = static_cast<double>(i+1)/static_cast<double>(n_total_props_theta);
	logger->info("{:>4}  {:>6}        {:.2f}  {:g}", i+1, "θ", acc_theta, pm);
      }

      if(i >= n_burn)
      {
	blaze::column(theta_samples, i-n_burn) = theta;
	blaze::column(f_samples,     i-n_burn) = dist_q_f.mean;
	gram_samples[i-n_burn]                 = gram_chol;
      }
    }
    return {std::move(theta_samples),
      std::move(f_samples),
      std::move(gram_samples)};
  }
}

#endif
