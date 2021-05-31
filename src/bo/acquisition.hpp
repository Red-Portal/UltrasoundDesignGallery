
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

#ifndef __US_GALLERY_ACQUISITION_HPP__
#define __US_GALLERY_ACQUISITION_HPP__

#include "../gp/gp_prior.hpp"
#include "../math/blaze.hpp"
#include "../math/linearalgebra.hpp"
#include "../math/mvnormal.hpp"
#include "../system/debug.hpp"
#include "../system/profile.hpp"
#include "bayesian_optimization.hpp"
#include "cmaes.hpp"
#include "lbfgs.hpp"
#include "gp_inference.hpp"

#include "../../test/finitediff.hpp"

#include <vector>
#include <random>
#include <limits>

namespace usdg
{
  template <typename KernelType>
  std::tuple<double, blaze::DynamicVector<double>>
  thompson_xi_gradient(usdg::GP<KernelType> const& gp,
		       blaze::DynamicMatrix<double> const& data_mat,
		       size_t n_beta,
		       blaze::DynamicVector<double> const& x,
		       blaze::DynamicVector<double> const& xi,
		       bool derivative)
  {
    size_t n_dims = x.size();
    double N      = static_cast<double>(n_beta);
    auto grad_buf = [derivative, n_dims]{
      if(derivative)
      {
	return blaze::DynamicVector<double>(n_dims, 0.0);
      }
      else
      {
	return blaze::DynamicVector<double>();
      }}();
    auto [lb, ub, dlbdxi_idx, dlbdxi_val, dubdxi_idx, dubdxi_val] =
      usdg::dbounds_dxi(x, xi);
    auto beta_delta = (ub - lb)/(N-1);

    auto pred_sum = 0.0;
    for (size_t i = 0; i < n_beta; ++i)
    {
      auto beta        = lb + beta_delta*static_cast<double>(i);
      auto x_beta      = x + beta*xi;
      auto [mean, var] = gp.predict(data_mat, x_beta);
      pred_sum        += mean;

      if (derivative)
      {
	auto dmu_dx            = usdg::gradient_mean(gp, data_mat, x_beta);
	auto dmu_dxi           = blaze::evaluate(beta*dmu_dx);
	double dbetadxi_ub_val = dubdxi_val*(static_cast<double>(i))/(N-1);
	double dbetadxi_lb_val = dlbdxi_val*(1 - static_cast<double>(i)/(N-1));
	auto dmudxi_dot_xiunit = blaze::dot(dmu_dx, xi);

	dmu_dxi[dubdxi_idx]   += dbetadxi_ub_val*dmudxi_dot_xiunit;
	dmu_dxi[dlbdxi_idx]   += dbetadxi_lb_val*dmudxi_dot_xiunit;
	grad_buf += dmu_dxi;
      }
    }
    if (derivative)
    {
      grad_buf = grad_buf/N;
    }
    return { pred_sum/N, grad_buf };
  }

  template <typename KernelType>
  std::tuple<double, blaze::DynamicVector<double>>
  ei_x_gradient(usdg::GP<KernelType> const& gp,
		blaze::DynamicMatrix<double> const& data_mat,
		double y_opt,
		blaze::DynamicVector<double> const& x,
		bool derivative)
  {
    size_t n_dims = x.size();
    auto grad_buf = [derivative, n_dims]{
      if(derivative)
      {
	return blaze::DynamicVector<double>(n_dims, 0.0);
      }
      else
      {
	return blaze::DynamicVector<double>();
      }}();

    double ei = 0.0;
    if (derivative)
    {
      // T1: d\mu/dx(x) \Phi(z)
      // T2: \phi(z)((\mu - y*)dz/dx + 1/(2\sigma) d\sigma^2/dx) 
      // T3: \sigma d\phi/dx(z) dz/dxz

      auto [mean, var, dmu_dx, dvar_dx] = usdg::gradient_mean_var(gp, data_mat, x);

      double sigma = sqrt(var);
      double delta = (mean - y_opt);
      double z     = delta/sigma;
      double Phiz  = usdg::normal_cdf(z);
      double phiz  = usdg::dnormal(z);
      ei           = delta*Phiz + phiz*sigma;

      auto dz_dx   = dmu_dx / sigma;
      auto dphi_dz = usdg::gradient_dnormal(z)*dz_dx;
      auto t1      = dmu_dx*Phiz;
      auto t2      = phiz*(delta*dz_dx + 1.0/(2*sigma)*dvar_dx);
      auto t3      = sigma*dphi_dz;
      auto dei_dx  = t1 + t2 + t3;
      grad_buf += dei_dx;
    }
    else
    {
      auto [mean, var] = gp.predict(data_mat, x);
      double sigma     = sqrt(var);
      double delta     = (mean - y_opt);
      double z         = delta/sigma;
      double Phiz      = usdg::normal_cdf(z);
      double phiz      = usdg::dnormal(z);
      ei               = delta*Phiz + phiz*sigma;
    }
    return { ei, grad_buf };
  }

  class ExpectedImprovementKoyama {};
  class ExpectedImprovement {};

  template <typename KernelType>
  inline std::pair<blaze::DynamicVector<double>, double>
  find_best_alpha(usdg::Dataset const& data,
		  blaze::DynamicMatrix<double> const& data_mat,
		  usdg::GP<KernelType> const& gp)
  {
    double y_opt = std::numeric_limits<double>::lowest(); 
    auto x_opt   = blaze::DynamicVector<double>();
    for (size_t i = 0; i < data.num_data(); ++i)
    {
      auto x_alpha     = blaze::column(data_mat, data.alpha_index(i));
      auto [mean, var] = gp.predict(data_mat, x_alpha);
      if (mean > y_opt)
      {
	y_opt = mean;
	x_opt = x_alpha;
      }
    }
    return { std::move(x_opt), y_opt };
  }

  // template <>
  // template <typename Rng, typename BudgetType>
  // inline std::tuple<blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>>
  // BayesianOptimization<usdg::ExpectedImprovement>::
  // next_query(Rng& prng,
  // 	     size_t n_burn,
  // 	     size_t n_samples,
  // 	     BudgetType budget,
  // 	     usdg::MvNormal<usdg::DiagonalChol> const& prior_dist,
  // 	     usdg::Profiler* profiler,
  // 	     spdlog::logger* logger) const
  // {
  //   if(logger)
  //   {
  //     logger->info("Finding next Bayesian optimization query with expected improvement: {}",
  // 		   usdg::file_name(__FILE__));
  //   }
  //   if(profiler)
  //   {
  //     profiler->start("next_query"s);
  //     profiler->start("sample_gp_hyper"s);
  //   }

  //   auto data_mat = this->_data.data_matrix();
  //   auto [theta_samples, f_samples, K_samples] = usdg::sample_gp_hyper(
  //     prng,
  //     this->_data,
  //     data_mat,
  //     n_burn,
  //     n_samples,
  //     prior_dist.mean,
  //     prior_dist,
  //     nullptr);
  //   if(profiler)
  //   {
  //     profiler->stop("sample_gp_hyper"s);
  //     profiler->start("optimize_acquisition"s);
  //   }

  //   auto mgp = usdg::MarginalizedGP<usdg::SquaredExpIso>(blaze::exp(theta_samples),
  // 							 f_samples,
  // 							 K_samples);

  //   auto [_, y_opt] = usdg::find_best_alpha(this->_data, data_mat, mgp);
  //   size_t n_dims   = this->_n_dims;
  //   auto ei_x_acq = [&](blaze::DynamicVector<double> const& x_in) {
  //     auto [mean, var] = mgp.predict(data_mat, x_in);
  //     double sigma     = sqrt(var);
  //     double delta     = (mean - y_opt);
  //     double ei        = delta*usdg::normal_cdf(delta/sigma)
  // 	+ sigma*usdg::dnormal(delta/sigma);
  //     return -ei;
  //   };
  //   auto [x_champ, y_x_champ] = usdg::cmaes_optimize(
  //     prng, ei_x_acq, n_dims, budget/2, logger);

  //   auto ei_xi_acq = [&](blaze::DynamicVector<double> const& xi_in) {
  //     size_t n_pseudo    = 16;
  //     auto [lb, ub]      = usdg::pbo_find_bounds(x_champ, xi_in);
  //     double beta_delta  = (ub - lb) / static_cast<double>(n_pseudo);
  //     double ei_avg      = 0.0;
  //     for (size_t i = 0; i < n_pseudo; ++i)
  //     {
  // 	auto beta        = lb + beta_delta*static_cast<double>(i);
  // 	auto [mean, var] = mgp.predict(data_mat, x_champ + beta*xi_in);
  // 	double sigma     = sqrt(var);
  // 	double delta     = (mean - y_opt);
  // 	double ei        = delta*usdg::normal_cdf(delta/sigma)
  // 	  + sigma*usdg::dnormal(delta/sigma);
  // 	ei_avg += ei;
  //     }
  //     return -ei_avg / static_cast<double>(n_pseudo);
  //   };
  //   auto [xi_champ, y_xi_champ] = usdg::cmaes_maxball_optimize(
  //     prng, ei_xi_acq, n_dims, budget/2, logger);

  //   std::cout << x_champ << std::endl;
  //   std::cout << xi_champ << std::endl;

  //   if(profiler)
  //   {
  //     profiler->stop("optimize_acquisition"s);
  //     profiler->stop("next_query"s);
  //   }
  //   if(logger)
  //   {
  //     logger->info("Found next Bayesian optimization query.");
  //   }
  //   return { std::move(x_champ), std::move(xi_champ) };
  // }

  template <>
  template <typename Rng, typename BudgetType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
  BayesianOptimization<usdg::ExpectedImprovementKoyama>::
  next_query(Rng& prng,
	     BudgetType budget,
	     blaze::DynamicVector<double> const& linescales,
	     usdg::Profiler* profiler,
	     spdlog::logger* logger) const
  {
    if(logger)
    {
      logger->info("Finding next Bayesian optimization query with expected improvement and Koyama scheme: {}",
		   usdg::file_name(__FILE__));
    }
    if(profiler)
    {
      profiler->start("next_query"s);
    }

    auto data_mat = this->_data.data_matrix();
    auto gp       = fit_gp(prng, this->_data, data_mat, linescales, logger);

    if(profiler)
    {
      profiler->stop("sample_gp_hyper"s);
      profiler->start("optimize_acquisition"s);
    }

    auto [x_opt, y_opt] = usdg::find_best_alpha(this->_data, data_mat, gp);

    auto ei_x_acq = [&](blaze::DynamicVector<double> const& x,
			bool with_gradient)
      -> std::pair<double, blaze::DynamicVector<double>>
      {
	auto [mean,var] = ei_x_gradient(gp, data_mat, y_opt, x, with_gradient);
	return {mean, var};
      };

    size_t n_dims   = this->_n_dims;
    auto [x_next, _] = usdg::lbfgs_multistage_box_optimize(
      prng, ei_x_acq, this->_n_dims, budget, 8, logger);

    auto delta   = x_opt - x_next;
    auto xi_next = delta / blaze::max(blaze::abs(delta));

    std::cout << x_next << std::endl;
    std::cout << xi_next<< std::endl;

    if(profiler)
    {
      profiler->stop("optimize_acquisition"s);
      profiler->stop("next_query"s);
    }
    if(logger)
    {
      logger->info("Found next Bayesian optimization query.");
    }
    return { std::move(x_opt), std::move(xi_next) };
  }
}

#endif
