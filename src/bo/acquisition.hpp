
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
#include "../gp/sample_beta.hpp"
#include "../math/blaze.hpp"
#include "../math/linearalgebra.hpp"
#include "../math/mvnormal.hpp"
#include "../math/uniform.hpp"
#include "../math/root.hpp"
#include "../system/debug.hpp"
#include "../system/profile.hpp"
#include "bayesian_optimization.hpp"
#include "lbfgs.hpp"
#include "fit_gp.hpp"

#include <vector>
#include <random>
#include <limits>

namespace usdg
{
  inline std::tuple<double, double, double, double, double, double>
  expected_improvement_precompute(double mean,
				  double var,
				  double y_opt)
  {
    double sigma = sqrt(var);
    double delta = (mean - y_opt);
    double z     = delta/sigma;
    double Phiz  = usdg::normal_cdf(z);
    double phiz  = usdg::dnormal(z);
    double ei    = delta*Phiz + phiz*sigma;
    return {ei, sigma, delta, z, Phiz, phiz};
  }

  inline double
  expected_improvement(double mean, double var, double y_opt)
  {
    [[maybe_unused]] auto [ei, sigma, delta, z, Phiz, phiz] =
      usdg::expected_improvement_precompute(mean, var, y_opt);
    return ei;
  }

  inline decltype(auto)
  gradient_expected_improvement(double sigma,
				double delta,
				double z,
				double Phiz,
				double phiz,
				blaze::DynamicVector<double> const& dmu_dx,
				blaze::DynamicVector<double> const& dvar_dx)
  {
      // T1: d\mu/dx(x) \Phi(z)
      // T2: \phi(z)((\mu - y*)dz/dx + 1/(2\sigma) d\sigma^2/dx) 
      // T3: \sigma d\phi/dx(z) dz/dxz

      auto dz_dx   = dmu_dx / sigma;
      auto dphi_dz = usdg::gradient_dnormal(z)*dz_dx;
      auto t1      = dmu_dx*Phiz;
      auto t2      = phiz*(delta*dz_dx + 1.0/(2*sigma)*dvar_dx);
      auto t3      = sigma*dphi_dz;
      return t1 + t2 + t3;
  }

  template <typename KernelType>
  std::tuple<double, blaze::DynamicVector<double>>
  ei_with_deidx(usdg::GP<KernelType> const& gp,
		blaze::DynamicMatrix<double> const& data_mat,
		double y_opt,
		blaze::DynamicVector<double> const& x,
		bool derivative)
  {
    size_t n_dims = x.size();
    auto grad_buf = [derivative, n_dims]{
      if(derivative)
	return blaze::DynamicVector<double>(n_dims, 0.0);
      else
	return blaze::DynamicVector<double>();
      }();

    double ei_res = 0.0;
    if (derivative)
    {
      auto [mean, var, dmudx, dvardx]        = usdg::gradient_mean_var(gp, data_mat, x);
      auto [ei, sigma, delta, z, Phiz, phiz] = usdg::expected_improvement_precompute(mean, var, y_opt);
      auto dei_dx = usdg::gradient_expected_improvement(sigma, delta, z, Phiz, phiz, dmudx, dvardx);
      ei_res    = ei;
      grad_buf += dei_dx;
    }
    else
    {
      auto [mean, var] = gp.predict(data_mat, x);
      double ei        = usdg::expected_improvement(mean, var, y_opt);
      ei_res           = ei;
    }
    return { ei_res, grad_buf };
  }

  // template <typename Rng,
  // 	    typename KernelType,
  // 	    typename XVec,
  // 	    typename XiVec>
  // inline double
  // approximate_expected_improvement(Rng& prng,
  // 				   usdg::GP<KernelType> const& gp,
  // 				   blaze::DynamicMatrix<double> const& data_mat,
  // 				   size_t n_beta,
  // 				   size_t n_montecarlo,
  // 				   size_t iter,
  // 				   double y_opt,
  // 				   XVec const& x,
  // 				   XiVec const& xi)
  // {
  //   auto unit_normal = std::normal_distribution<double>();
  //   size_t n_dims    = x.size();
  //   auto [lb, ub]    = usdg::pbo_find_bounds(x, xi);
  //   double ei_sum    = 0.0;
  //   auto x_betas     = blaze::DynamicMatrix<double>(x.size(), n_beta);
  //   for (size_t k = 0; k < n_montecarlo; ++k)
  //   {
  //     double y_beta_opt = std::numeric_limits<double>::lowest();
  //     auto betas        = usdg::sample_beta(prng, 0.0, lb, ub, iter, n_beta, n_dims);
  //     for (size_t j = 0; j < n_beta; ++j)
  //     {
  // 	blaze::column(x_betas, j) = x + betas[j]*xi;
  //     }
  //     auto [pred_means, pred_vars] = gp.predict(data_mat, x_betas);
  //     for (size_t j = 0; j < n_beta; ++j)
  //     {
  // 	double y_beta = unit_normal(prng)*sqrt(pred_vars[j]) + pred_means[j];
  // 	y_beta_opt    = std::max(y_beta, y_beta_opt);
  //     }
  //     ei_sum += std::max(y_beta_opt - y_opt, 0.0);
  //   }
  //   return ei_sum / static_cast<double>(n_montecarlo);
  // }

  // inline std::pair<double, double>
  // gumbel_quantile_matching(blaze::DynamicVector<double> means,
  // 			   blaze::DynamicVector<double> vars,
  // 			   double y_opt)
  // {
  //   auto stds = sqrt(vars);
  //   /* Gumbel quantile matching */
  //   auto emp_cdf = [&, means=means](double x){
  //     double product = 1;
  //     for (size_t i = 0; i < stds.size(); ++i)
  // 	product *= usdg::normal_cdf((means[i] - x)/stds[i]);
  //     return product;
  //   };

  //   double left  = y_opt;
  //   if(emp_cdf(left) < 0.25)
  //   {
  //     double right = blaze::max(means + 5*stds);
  //     while(emp_cdf(right) < 0.75)
  //     {
  // 	right += right - left;
  //     }
  //     double md = usdg::find_zero(left, right, 0.01,
  // 				  [&emp_cdf](double x_in){
  // 				    return emp_cdf(x_in) - 0.5;
  // 				  });
  //     double q1 = usdg::find_zero(left,    md, 0.01,
  // 				  [&emp_cdf](double x_in){
  // 				    return emp_cdf(x_in) - 0.25;
  // 				  });
  //     double q2 = usdg::find_zero(  md, right, 0.01,
  // 				    [&emp_cdf](double x_in){
  // 				      return emp_cdf(x_in) - 0.75;
  // 				    });

  //     double gumbel_beta = (q1 - q2)/ (log(log(4.0/3)) - log(log(4.0)));
  //     double gumbel_mu   = md + gumbel_beta*log(log(2));
  //     return {gumbel_mu, gumbel_beta};
  //   }
  // }

  // template <typename Rng,
  // 	    typename KernelType,
  // 	    typename XVec,
  // 	    typename XiVec>
  // inline double
  // approximate_expected_improvement(Rng& prng,
  // 				   usdg::GP<KernelType> const& gp,
  // 				   blaze::DynamicMatrix<double> const& data_mat,
  // 				   size_t n_beta,
  // 				   size_t,
  // 				   size_t iter,
  // 				   double y_opt,
  // 				   XVec const& x,
  // 				   XiVec const& xi)
  // {
  //   auto unit_normal = std::normal_distribution<double>();
  //   size_t n_dims    = x.size();
  //   auto [lb, ub]    = usdg::pbo_find_bounds(x, xi);
  //   double ei_sum    = 0.0;
  //   auto x_betas     = blaze::DynamicMatrix<double>(x.size(), n_beta);

  //   auto betas = usdg::sample_beta(prng, 0.0, lb, ub, iter, n_beta, n_dims);
  //   for (size_t j = 0; j < n_beta; ++j)
  //   {
  //     blaze::column(x_betas, j) = x + betas[j]*xi;
  //   }
  //   auto [pred_means, pred_vars] = gp.predict(data_mat, x_betas);

  //   auto [gumbel_mu, gumbel_beta] = usdg::gumbel_quantile_matching(pred_means,
  // 								   pred_vars,
  // 								   y_opt);
  //   /* Shifting Gumbel mean since truncation is assumed at x = 0 */
  //   double gumbel_mu_truncated = gumbel_mu - y_opt;
    
  //   double gumbel_lb = 0.0;
  //   double gumbel_ub = exp(gumbel_mu / gumbel_beta);
  //   double gamma       = usdg::gauss_legendre([=](double y){
  //     if(y < gumbel_lb || y > exp(gumbel_ub))
  // 	return 0.0;
  //     else
  // 	return log(y);
  //   });
  //   double gumbel_truncated_mean = gumbel_mu_truncated
  //     - (gumbel_beta*gamma) / (1 - exp(-exp(gumbel_mu_truncated / gumbel_beta)));
  //   double gumbel_truncated_cdf = 1 - exp(- exp(- (y_opt - gumbel_truncated_mean) / gumbel_beta));
  //   return gumbel_truncated_mean - y_opt*gumbel_truncated_cdf;

  //   // for (size_t k = 0; k < n_montecarlo; ++k)
  //   // {
  //   //   double y_beta_opt = std::numeric_limits<double>::lowest();
  //   //   for (size_t j = 0; j < n_beta; ++j)
  //   //   {
  //   // 	double y_beta = unit_normal(prng)*sqrt(pred_vars[j]) + pred_means[j];
  //   // 	y_beta_opt    = std::max(y_beta, y_beta_opt);
  //   //   }
  //   //   ei_sum += std::max(y_beta_opt - y_opt, 0.0);
  //   // }
  //   // return ei_sum / static_cast<double>(n_montecarlo);
  // }

  // template <typename Rng,
  // 	    typename KernelType>
  // inline blaze::DynamicVector<double>
  // find_xi_ei_random(Rng& prng,
  // 		    usdg::GP<KernelType> const& gp,
  // 		    blaze::DynamicMatrix<double> const& data_mat,
  // 		    size_t n_beta,
  // 		    size_t n_mc,
  // 		    size_t iter,
  // 		    double y_opt,
  // 		    blaze::DynamicVector<double> const& x,
  // 		    blaze::DynamicVector<double> const& xi_init,
  // 		    size_t n_eval)
  // {
  //   size_t n_dims = xi_init.size();
  //   auto xi_opt   = blaze::DynamicVector<double>(n_dims);
  //   auto xi       = blaze::DynamicVector<double>(n_dims);
  //   auto ei_opt   = std::numeric_limits<double>::lowest();

  //   for (size_t i = 0; i < n_eval; ++i)
  //   {
  //     double beta_range = 0.0;
  //     do
  //     {
  // 	xi  = usdg::rmvnormal(prng, n_dims);
  // 	xi /= blaze::max(blaze::abs(xi));
  // 	auto [ub, lb] = usdg::pbo_find_bounds(x, xi);
  // 	beta_range    = abs(ub - lb);
  //     }
  //     while (beta_range < 1e-4);

  //     auto ei = usdg::approximate_expected_improvement(prng, gp, data_mat,
  // 						       n_beta, n_mc, iter,
  // 						       y_opt, x, xi);
  //     if(ei > ei_opt)
  //     {
  // 	ei_opt = ei;
  // 	xi_opt = xi;
  //     }
  //   }
  //   return xi_opt;
  // }

  template <typename Rng,
	    typename KernelType>
  inline std::pair<blaze::DynamicVector<double>, double>
  find_best_x_lbfgs(Rng& prng,
		    blaze::DynamicMatrix<double> const& data_mat,
		    usdg::GP<KernelType> const& gp,
		    size_t n_dims,
		    size_t budget,
		    spdlog::logger* logger)
  {
    auto mean_with_grad = [&](blaze::DynamicVector<double> const& x, bool)
      -> std::pair<double, blaze::DynamicVector<double>>
    {
      auto [mean, dmudx] = usdg::gradient_mean(gp, data_mat, x);
      return {mean, dmudx};
    };

    size_t n_restarts = 4;
    return usdg::lbfgs_multistage_box_optimize(
      prng, mean_with_grad, n_dims, budget/n_restarts, n_restarts, logger);
  }


  template <typename Rng,
	    typename KernelType>
  inline blaze::DynamicVector<double>
  find_x_ei_lbfgs(Rng& prng,
		  usdg::GP<KernelType> const& gp,
		  blaze::DynamicMatrix<double> const& data_mat,
		  double y_opt,
		  size_t n_dims,
		  size_t budget,
		  spdlog::logger* logger)
  {
    auto ei_x_acq = [&](blaze::DynamicVector<double> const& x,
			bool with_gradient)
      -> std::pair<double, blaze::DynamicVector<double>>
      {
	auto [ei, deidx] = ei_with_deidx(gp, data_mat, y_opt, x, with_gradient);
	return {ei, deidx};
      };

    size_t n_restarts = 4;
    auto [x_next, _]  = usdg::lbfgs_multistage_box_optimize(
      prng, ei_x_acq, n_dims, budget/n_restarts, n_restarts, logger);
    return x_next;
  }

  // class AEI_AEI {};

  // template <>
  // template <typename Rng, typename BudgetType>
  // inline std::tuple<blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>,
  // 		    double>
  // BayesianOptimization<usdg::AEI_AEI>::
  // next_query(Rng& prng,
  // 	     size_t iter,
  // 	     size_t n_pseudo,
  // 	     BudgetType budget,
  // 	     blaze::DynamicVector<double> const& linescales,
  // 	     usdg::Profiler* profiler,
  // 	     spdlog::logger* logger) const
  // {
  //   if(logger)
  //   {
  //     logger->info("Finding next Bayesian optimization query with approximate expected improvement: {}",
  // 		   usdg::file_name(__FILE__));
  //   }
  //   if(profiler)
  //   {
  //     profiler->start("next_query"s);
  //     profiler->start("fit_gp"s);
  //   }

  //   auto data_mat = this->_data.data_matrix();
  //   auto gp       = fit_gp(prng, this->_data, data_mat, linescales, logger);

  //   if(profiler)
  //   {
  //     profiler->stop("fit_gp"s);
  //     profiler->start("optimize_acquisition"s);
  //   }

  //   size_t n_dims       = this->_n_dims;
  //   auto [x_opt, y_opt] = usdg::find_best_x_lbfgs(prng, data_mat, gp,
  // 						  n_dims, budget, logger);

  //   size_t n_montecarlo = 32;
  //   auto obj = [&,y_opt=y_opt](blaze::DynamicVector<double> const& x_xi)
  //     -> double
  //   {
  //     auto x  = blaze::subvector(x_xi, 0,      n_dims);
  //     auto xi = blaze::subvector(x_xi, n_dims, n_dims);
  //     return usdg::approximate_expected_improvement(
  // 	prng, gp, data_mat, n_pseudo, n_montecarlo, iter, y_opt, x, xi);
  //   };

  //   auto proj = [&](blaze::DynamicVector<double> const& x_xi) {
  //     auto x_xi_res = blaze::DynamicVector<double>(x_xi.size());
  //     auto x        = blaze::subvector(x_xi, 0,      n_dims);
  //     auto xi       = blaze::subvector(x_xi, n_dims, n_dims);

  //     auto x_feasible  = blaze::clamp(x, 0.0, 1.0);
  //     auto xi_feasible = xi / blaze::max(blaze::abs(xi));
  //     blaze::subvector(x_xi_res, 0,      n_dims) = x_feasible;
  //     blaze::subvector(x_xi_res, n_dims, n_dims) = xi_feasible;
  //     return x_xi_res;
  //   };

  //   auto xi_init = usdg::rmvnormal(prng, n_dims);
  //   xi_init     /= blaze::max(blaze::abs(xi_init));

  //   auto x_xi_init  = blaze::DynamicVector<double>(n_dims*2);
  //   blaze::subvector(x_xi_init, 0,      n_dims) = x_opt;
  //   blaze::subvector(x_xi_init, n_dims, n_dims) = xi_init;

  //   double noise_sd = 1.0;
  //   double stepsize = 1.0/static_cast<double>(n_dims);
  //   auto x_xi_next = usdg::spsa_maximize(prng, obj, proj, noise_sd, stepsize,
  // 					 x_xi_init, budget/n_montecarlo/2);
  //   auto x_next    = blaze::subvector(x_xi_next, 0,      n_dims);
  //   auto xi_next   = blaze::subvector(x_xi_next, n_dims, n_dims);

  //   if(profiler)
  //   {
  //     profiler->stop("next_query"s);
  //   }
  //   if(logger)
  //   {
  //     logger->info("Found next Bayesian optimization query.");
  //   }
  //   return { x_next, xi_next, std::move(x_opt), y_opt };
  // }

  // class EI_AEI {};

  // template <>
  // template <typename Rng, typename BudgetType>
  // inline std::tuple<blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>,
  // 		    double>
  // BayesianOptimization<usdg::EI_AEI>::
  // next_query(Rng& prng,
  // 	     size_t iter,
  // 	     size_t n_pseudo,
  // 	     BudgetType budget,
  // 	     blaze::DynamicVector<double> const& linescales,
  // 	     usdg::Profiler* profiler,
  // 	     spdlog::logger* logger) const
  // {
  //   if(logger)
  //   {
  //     logger->info("Finding next Bayesian optimization query with expected improvement and random search: {}",
  // 		   usdg::file_name(__FILE__));
  //   }
  //   if(profiler)
  //   {
  //     profiler->start("next_query"s);
  //     profiler->start("fit_gp"s);
  //   }

  //   auto data_mat = this->_data.data_matrix();
  //   auto gp       = fit_gp(prng, this->_data, data_mat, linescales, logger);

  //   if(profiler)
  //   {
  //     profiler->stop("fit_gp"s);
  //     profiler->start("optimize_acquisition"s);
  //   }

  //   size_t n_dims       = this->_n_dims;
  //   auto [x_opt, y_opt] = usdg::find_best_x_lbfgs(prng, data_mat, gp,
  // 						  n_dims, budget, logger);
  //   auto x_next         = usdg::find_x_ei_lbfgs(prng, gp, data_mat, y_opt,
  // 						this->_n_dims, budget/2, logger);
  //   auto delta     = x_opt - x_next;
  //   auto xi_koyama = delta / blaze::max(blaze::abs(delta));

  //   size_t n_montecarlo = 32;
  //   size_t n_eval       = budget / n_montecarlo / 2;
  //   auto xi_next        = usdg::find_xi_ei_random(prng, gp, data_mat, n_pseudo,
  // 						  n_montecarlo, iter, y_opt,
  // 						  x_next, xi_koyama, n_eval);
  //   if(profiler)
  //   {
  //     profiler->stop("optimize_acquisition"s);
  //     profiler->stop("next_query"s);
  //   }
  //   if(logger)
  //   {
  //     logger->info("Found next Bayesian optimization query.");
  //   }
  //   return { std::move(x_next), std::move(xi_next), std::move(x_opt), y_opt };
  // }

  // class EI_Random {};

  // template <>
  // template <typename Rng, typename BudgetType>
  // inline std::tuple<blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>,
  // 		    double>
  // BayesianOptimization<usdg::EI_Random>::
  // next_query(Rng& prng,
  // 	     size_t,
  // 	     size_t,
  // 	     BudgetType budget,
  // 	     blaze::DynamicVector<double> const& linescales,
  // 	     usdg::Profiler* profiler,
  // 	     spdlog::logger* logger) const
  // {
  //   if(logger)
  //   {
  //     logger->info("Finding next Bayesian optimization query with expected improvement and a Random direction: {}",
  // 		   usdg::file_name(__FILE__));
  //   }
  //   if(profiler)
  //   {
  //     profiler->start("next_query"s);
  //     profiler->start("fit_gp"s);
  //   }

  //   auto data_mat = this->_data.data_matrix();
  //   auto gp       = fit_gp(prng, this->_data, data_mat, linescales, logger);

  //   if(profiler)
  //   {
  //     profiler->stop("fit_gp"s);
  //     profiler->start("optimize_acquisition"s);
  //   }

  //   size_t n_dims       = this->_n_dims;
  //   auto [x_opt, y_opt] = usdg::find_best_x_lbfgs(prng, data_mat, gp,
  // 						  n_dims, budget, logger);
  //   auto x_next         = usdg::find_x_ei_lbfgs(prng, gp, data_mat, y_opt,
  // 						this->_n_dims, budget, logger);
  //   /* sample random direction for xi with a feasible range for beta */
  //   auto xi_next = blaze::DynamicVector<double>(this->_n_dims);
  //   double beta_range = 0.0;
  //   do
  //   {
  //     xi_next  = usdg::rmvnormal(prng, this->_n_dims);
  //     xi_next /= blaze::max(blaze::abs(xi_next));
  //     auto [ub, lb] = usdg::pbo_find_bounds(x_next, xi_next);
  //     beta_range    = abs(ub - lb);
  //   }
  //   while (beta_range < 1e-4);

  //   if(profiler)
  //   {
  //     profiler->stop("optimize_acquisition"s);
  //     profiler->stop("next_query"s);
  //   }
  //   if(logger)
  //   {
  //     logger->info("Found next Bayesian optimization query.");
  //   }
  //   return { std::move(x_next), std::move(xi_next), std::move(x_opt), y_opt  };
  // }

  class EI_Koyama {};

  template <>
  template <typename Rng, typename BudgetType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    double>
  BayesianOptimization<usdg::EI_Koyama>::
  next_query(Rng& prng,
	     size_t,
	     size_t,
	     BudgetType budget,
	     blaze::DynamicVector<double> const& linescales,
	     usdg::Profiler* profiler,
	     spdlog::logger* logger) const
  {
    if(logger)
    {
      logger->info("Finding next Bayesian optimization query with expected improvement and the Koyama scheme: {}",
		   usdg::file_name(__FILE__));
    }
    if(profiler)
    {
      profiler->start("next_query"s);
      profiler->start("fit_gp"s);
    }

    auto data_mat = this->_data.data_matrix();
    auto gp       = fit_gp(prng, this->_data, data_mat, linescales, logger);

    if(profiler)
    {
      profiler->stop("fit_gp"s);
      profiler->start("optimize_acquisition"s);
    }

    size_t n_dims       = this->_n_dims;
    auto [x_opt, y_opt] = usdg::find_best_x_lbfgs(prng, data_mat, gp,
						  n_dims, budget, logger);
    auto x_next         = usdg::find_x_ei_lbfgs(prng, gp, data_mat, y_opt,
						this->_n_dims, budget, logger);

    auto delta   = x_opt - x_next;
    auto xi_next = blaze::evaluate(delta / blaze::max(blaze::abs(delta)));

    if(profiler)
    {
      profiler->stop("optimize_acquisition"s);
      profiler->stop("next_query"s);
    }
    if(logger)
    {
      logger->info("Found next Bayesian optimization query.");
    }
    return { std::move(x_next), std::move(xi_next), std::move(x_opt), y_opt };
  }

  // class PCD {};

  // template <>
  // template <typename Rng, typename BudgetType>
  // inline std::tuple<blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>,
  // 		    double>
  // BayesianOptimization<usdg::PCD>::
  // next_query(Rng& prng,
  // 	     size_t,
  // 	     size_t,
  // 	     BudgetType budget,
  // 	     blaze::DynamicVector<double> const& linescales,
  // 	     usdg::Profiler* profiler,
  // 	     spdlog::logger* logger) const
  // {
  //   if(logger)
  //   {
  //     logger->info("Finding next Bayesian optimization query with expected improvement and the Koyama scheme: {}",
  // 		   usdg::file_name(__FILE__));
  //   }
  //   if(profiler)
  //   {
  //     profiler->start("next_query"s);
  //     profiler->start("fit_gp"s);
  //   }

  //   auto data_mat = this->_data.data_matrix();
  //   auto gp       = fit_gp(prng, this->_data, data_mat, linescales, logger);

  //   if(profiler)
  //   {
  //     profiler->stop("fit_gp"s);
  //     profiler->start("optimize_acquisition"s);
  //   }

  //   size_t n_dims       = this->_n_dims;
  //   auto [x_opt, y_opt] = usdg::find_best_x_lbfgs(prng, data_mat, gp,
  // 						  n_dims, budget, logger);
  //   // auto x_next         = usdg::find_x_ei_lbfgs(prng, gp, data_mat, y_opt,
  //   // 						this->_n_dims, budget, logger);

  //   // auto delta   = x_opt - x_next;
  //   // auto xi_next = blaze::evaluate(delta / blaze::max(blaze::abs(delta)));

  //   if(profiler)
  //   {
  //     profiler->stop("optimize_acquisition"s);
  //     profiler->stop("next_query"s);
  //   }
  //   if(logger)
  //   {
  //     logger->info("Found next Bayesian optimization query.");
  //   }
  //   return { std::move(x_next), std::move(xi_next), std::move(x_opt), y_opt };
  // }
}

#endif
