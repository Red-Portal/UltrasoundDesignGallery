
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
#include "../system/debug.hpp"
#include "../system/profile.hpp"
#include "bayesian_optimization.hpp"
#include "lbfgs.hpp"
#include "spsa.hpp"
#include "gp_inference.hpp"

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

  template <typename Rng,
	    typename KernelType>
  inline double
  projective_expected_improvement(Rng& prng,
				  usdg::GP<KernelType> const& gp,
				  blaze::DynamicMatrix<double> const& data_mat,
				  size_t n_beta,
				  size_t iter,
				  double y_opt,
				  blaze::DynamicVector<double> const& x,
				  blaze::DynamicVector<double> const& xi)
  {
    size_t n_dims = x.size();
    auto grad_buf = blaze::DynamicVector<double>(n_dims, 0.0);

    auto [lb, ub] = usdg::pbo_find_bounds(x, xi);
    auto betas    = usdg::rmvuniform(prng, n_beta, lb, ub);
    auto p_beta   = usdg::beta_pdf(0.0, lb, ub, iter, n_dims);
    auto q_beta   = 1.0 / (ub - lb);
    double ei_sum = 0.0;
    for (double beta_i : betas)
    {
      auto x_beta      = x + beta_i*xi;
      auto [mean, var] = gp.predict(data_mat, x_beta);
      auto ei          = usdg::expected_improvement(mean, var, y_opt);
      ei_sum          += ei * p_beta(beta_i) / q_beta;
    }
    return ei_sum / static_cast<double>(n_beta);
  }

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

  template <typename Rng,
	    typename KernelType>
  inline blaze::DynamicVector<double>
  find_xi_ei_random(Rng& prng,
		    usdg::GP<KernelType> const& gp,
		    blaze::DynamicMatrix<double> const& data_mat,
		    size_t n_beta,
		    size_t iter,
		    double y_opt,
		    blaze::DynamicVector<double> const& x,
		    blaze::DynamicVector<double> const& xi_init,
		    size_t n_eval)
  {
    size_t n_dims = xi_init.size();
    auto xi_opt   = blaze::DynamicVector<double>(n_dims);
    auto xi       = blaze::DynamicVector<double>(n_dims);
    auto ei_opt   = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < n_eval; ++i)
    {
      double beta_range = 0.0;
      do
      {
	xi = usdg::rmvnormal(prng, n_dims);
	xi /= blaze::max(blaze::abs(xi));
	auto [ub, lb] = usdg::pbo_find_bounds(x, xi);
	beta_range    = abs(ub - lb);
      }
      while (beta_range < 1e-4);

      auto ei = projective_expected_improvement(prng, gp, data_mat, n_beta,
						iter, y_opt, x, xi);
      if(ei > ei_opt)
      {
	ei_opt = ei;
	xi_opt = xi;
      }
    }
    return xi_opt;
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

    size_t n_restarts = 8;
    auto [x_next, _]  = usdg::lbfgs_multistage_box_optimize(
      prng, ei_x_acq, n_dims, budget/n_restarts, n_restarts, logger);
    return x_next;
  }

  class ExpectedImprovementDTS {};

  template <>
  template <typename Rng, typename BudgetType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    double>
  BayesianOptimization<usdg::ExpectedImprovementDTS>::
  next_query(Rng& prng,
	     size_t iter,
	     size_t n_pseudo,
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

    size_t n_dims    = this->_n_dims;
    size_t K         = 16;
    size_t J         = 8;
    auto unit_normal = std::normal_distribution<double>();
    auto obj = [&](blaze::DynamicVector<double> const& x_xi) {
      auto x        = blaze::subvector(x_xi, 0,      n_dims);
      auto xi       = blaze::subvector(x_xi, n_dims, n_dims);
      auto [lb, ub] = usdg::pbo_find_bounds(x, xi);
      double ei_sum = 0.0;
      for (size_t k = 0; k < K; ++k)
      {
	auto betas        = usdg::sample_beta(prng, 0.0, lb, ub, iter, J, n_dims);
	double y_beta_opt = std::numeric_limits<double>::lowest();
	for (size_t j = 0; j < J; ++j)
	{
	  auto [pred_mean, pred_var] = gp.predict(data_mat, x + betas[j]*xi);
	  double y_beta = unit_normal(prng)*sqrt(pred_var) + pred_mean;
	  y_beta_opt    = std::max(y_beta, y_beta_opt);
	}
	ei_sum += std::max(y_beta_opt - y_opt, 0.0);
      }
      return ei_sum / K;
    };

    auto proj = [&](blaze::DynamicVector<double> const& x_xi) {
      auto x_xi_res = blaze::DynamicVector<double>(x_xi.size());
      auto x        = blaze::subvector(x_xi, 0,      n_dims);
      auto xi       = blaze::subvector(x_xi, n_dims, n_dims);

      auto x_feasible  = blaze::clamp(x, 0.0, 1.0);
      auto xi_feasible = xi / blaze::max(blaze::abs(xi));
      blaze::subvector(x_xi_res, 0,      n_dims) = x_feasible;
      blaze::subvector(x_xi_res, n_dims, n_dims) = xi_feasible;
      return x_xi_res;
    };

    auto xi_init = usdg::rmvnormal(prng, n_dims);
    xi_init     /= blaze::max(blaze::abs(xi_init));

    auto x_xi_init  = blaze::DynamicVector<double>(n_dims*2);
    blaze::subvector(x_xi_init, 0,      n_dims) = x_opt;
    blaze::subvector(x_xi_init, n_dims, n_dims) = xi_init;

    double noise_sd = 1.0;
    double stepsize = 0.01*sqrt(static_cast<double>(n_dims));
    auto x_xi_next = usdg::spsa_maximize(prng, obj, proj, noise_sd,
					 stepsize, x_xi_init, budget/K);
    auto x_next    = blaze::subvector(x_xi_next, 0,      n_dims);
    auto xi_next   = blaze::subvector(x_xi_next, n_dims, n_dims);

    if(profiler)
    {
      profiler->stop("optimize_acquisition"s);
      profiler->stop("next_query"s);
    }
    if(logger)
    {
      logger->info("Found next Bayesian optimization query.");
    }
    return { x_next, xi_next, std::move(x_opt), y_opt };
  }

  // class ExpectedImprovementRandom {};

  // template <>
  // template <typename Rng, typename BudgetType>
  // inline std::tuple<blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>>
  // BayesianOptimization<usdg::ExpectedImprovementRandom>::
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
  //     logger->info("Finding next Bayesian optimization query with expected improvement and Koyama scheme: {}",
  // 		   usdg::file_name(__FILE__));
  //   }
  //   if(profiler)
  //   {
  //     profiler->start("next_query"s);
  //   }

  //   auto data_mat = this->_data.data_matrix();
  //   auto gp       = fit_gp(prng, this->_data, data_mat, linescales, logger);

  //   if(profiler)
  //   {
  //     profiler->stop("sample_gp_hyper"s);
  //     profiler->start("optimize_acquisition"s);
  //   }

  //   auto [x_opt, y_opt] = usdg::find_best_alpha(this->_data, data_mat, gp);
  //   auto x_next         = usdg::find_x_ei_lbfgs(prng, gp, data_mat, y_opt,
  // 						this->_n_dims, budget, logger);
  //   auto delta     = x_opt - x_next;
  //   auto xi_koyama = delta / blaze::max(blaze::abs(delta));
  //   auto xi_next   = usdg::find_xi_ei_random(prng, gp, data_mat, n_pseudo, iter,
  // 					     y_opt, x_next, xi_koyama, budget);

  //   std::cout << x_next << std::endl;
  //   std::cout << xi_next<< std::endl;

  //   if(profiler)
  //   {
  //     profiler->stop("optimize_acquisition"s);
  //     profiler->stop("next_query"s);
  //   }
  //   if(logger)
  //   {
  //     logger->info("Found next Bayesian optimization query.");
  //   }
  //   return { std::move(x_next), std::move(xi_next) };
  // }

  class ExpectedImprovement {};

  template <>
  template <typename Rng, typename BudgetType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    double>
  BayesianOptimization<usdg::ExpectedImprovement>::
  next_query(Rng& prng,
	     size_t iter,
	     size_t n_pseudo,
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
    auto x_next         = usdg::find_x_ei_lbfgs(prng, gp, data_mat, y_opt,
						this->_n_dims, budget/2, logger);
    auto delta     = x_opt - x_next;
    auto xi_koyama = delta / blaze::max(blaze::abs(delta));
    auto xi_next   = usdg::find_xi_ei_random(prng, gp, data_mat, n_pseudo, iter,
					     y_opt, x_next, xi_koyama, budget/2);

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
    return { std::move(x_next), std::move(xi_next), std::move(x_opt), y_opt };
  }

  class ExpectedImprovementRandom {};

  template <>
  template <typename Rng, typename BudgetType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    double>
  BayesianOptimization<usdg::ExpectedImprovementRandom>::
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
      logger->info("Finding next Bayesian optimization query with expected improvement and Coordinate Projections: {}",
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
    auto x_next         = usdg::find_x_ei_lbfgs(prng, gp, data_mat, y_opt,
						this->_n_dims, budget, logger);
    /* sample random direction for xi with a feasible range for beta */
    auto xi_next = blaze::DynamicVector<double>(this->_n_dims);
    double beta_range = 0.0;
    do
    {
      xi_next  = usdg::rmvnormal(prng, this->_n_dims);
      xi_next /= blaze::max(blaze::abs(xi_next));
      auto [ub, lb] = usdg::pbo_find_bounds(x_next, xi_next);
      beta_range    = abs(ub - lb);
    }
    while (beta_range < 1e-4);

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
    return { std::move(x_next), std::move(xi_next), std::move(x_opt), y_opt  };
  }

  class ExpectedImprovementKoyama {};

  template <>
  template <typename Rng, typename BudgetType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    double>
  BayesianOptimization<usdg::ExpectedImprovementKoyama>::
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
    auto x_next         = usdg::find_x_ei_lbfgs(prng, gp, data_mat, y_opt,
						this->_n_dims, budget, logger);

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
    return { std::move(x_next), std::move(xi_next), std::move(x_opt), y_opt };
  }
}

#endif
