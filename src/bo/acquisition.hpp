
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
#include "sgd.hpp"
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
  inline std::tuple<double, blaze::DynamicVector<double>>
  ei_with_deidxi(usdg::GP<KernelType> const& gp,
		 blaze::DynamicMatrix<double> const& data_mat,
		 size_t n_beta,
		 double y_opt,
		 blaze::DynamicVector<double> const& x,
		 blaze::DynamicVector<double> const& xi,
		 bool derivative)
  {
    size_t n_dims = x.size();
    auto grad_buf = [derivative, n_dims]{
      if(derivative)
	return blaze::DynamicVector<double>(n_dims, 0.0);
      else
	return blaze::DynamicVector<double>();
    }();

    auto [lb, ub, dlbdxi_idx, dlbdxi_val, dubdxi_idx, dubdxi_val] =
      usdg::dbounds_dxi(x, xi);
    double N          = static_cast<double>(n_beta);
    double beta_delta = (ub - lb)/(N-1);
    double ei_sum     = 0.0;
    for (size_t i = 0; i < n_beta; ++i)
    {
      auto beta   = lb + beta_delta*static_cast<double>(i);
      auto x_beta = x + beta*xi;
      auto [mean, var, dmudxbeta, dvardxbeta] = usdg::gradient_mean_var(gp, data_mat, x_beta);
      auto [ei, sigma, delta, z, Phiz, phiz]  = usdg::expected_improvement_precompute(mean, var, y_opt);

      ei_sum += ei;

      if(derivative)
      {
	auto deidxbeta = usdg::gradient_expected_improvement(
	  sigma, delta, z, Phiz, phiz, dmudxbeta, dvardxbeta);
	auto deidxi    =  blaze::evaluate(beta*deidxbeta);

	double dbetadxi_ub_val = dubdxi_val*(static_cast<double>(i))/(N-1);
	double dbetadxi_lb_val = dlbdxi_val*(1 - static_cast<double>(i)/(N-1));
	auto deidxi_dot_xiunit = blaze::dot(deidxbeta, xi);

	deidxi[dubdxi_idx] += dbetadxi_ub_val*deidxi_dot_xiunit;
	deidxi[dlbdxi_idx] += dbetadxi_lb_val*deidxi_dot_xiunit;
	grad_buf           += deidxi;
      }
    }

    if(derivative)
    {
      grad_buf /= N;
    }

    return { ei_sum/N, grad_buf };
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

    auto [x_next, _] = usdg::lbfgs_multistage_box_optimize(
      prng, ei_x_acq, n_dims, budget, 8, logger);
    return x_next;
  }

  class ExpectedImprovement {};

  template <>
  template <typename Rng, typename BudgetType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
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
						this->_n_dims, budget, logger);
    auto delta     = x_opt - x_next;
    auto xi_koyama = delta / blaze::max(blaze::abs(delta));
    auto xi_next   = usdg::find_xi_ei_random(prng, gp, data_mat, n_pseudo, iter,
					     y_opt, x_next, xi_koyama, budget);


    // auto ei_xi_acq = [&](blaze::DynamicVector<double> const& xi_in,
    // 			 bool with_gradient)
    //   -> std::pair<double, blaze::DynamicVector<double>>
    //   {
    // 	auto [ei, deidxi] = ei_with_deidxi(gp, data_mat, n_beta, y_opt,
    // 					   x_next, xi_in, with_gradient);
    // 	return {ei, deidxi};
    //   };
    // auto [xi_next, __] = usdg::lbfgs_multistage_box_optimize(prng, ei_xi_acq,
    // 							     this->_n_dims,
    // 							     budget, n_restarts,
    // 							     logger);

    // auto deidxi = [&](blaze::DynamicVector<double> const& xi_in)
    //   -> blaze::DynamicVector<double>
    //   {
    // 	auto [__, deidx] = usdg::ei_with_deidxi(gp, data_mat, n_beta,
    // 						y_opt, x_next, xi_in, true);
    // 	deidx += usdg::rmvnormal(prng, deidx.size())*0.01;
    // 	return deidx;
    //   };

    // size_t iter = 1;
    // auto pei_xi  = [&](blaze::DynamicVector<double> const& xi_in)
    //   -> double
    // {
    //   // return usdg::projective_expected_improvement(prng, gp, data_mat, n_beta,

    //   // 						   iter, y_opt, x_next, xi_in);
    //   auto [ei, ___] = usdg::ei_with_deidxi(gp, data_mat, n_beta,
    // 					   y_opt, x_next, xi_in, false);
    //   return ei;
    // };

    // auto proj = [](blaze::DynamicVector<double> const& xi_in)
    // {
    //   return blaze::evaluate(xi_in / blaze::max(blaze::abs(xi_in)));
    // };

    // auto xi_init = proj(x_opt - x_next);
    // auto xi_next = usdg::sgd_maximize(deidxi,
    // 				      pei_xi,
    // 				      proj,
    // 				      xi_init,
    // 				      0.3,
    // 				      10000);
    // xi_next = proj(xi_next);
    //std::cout << pei_xi(xi_init) << " v.s. " << pei_xi(xi_next) << std::endl;

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
    return { std::move(x_next), std::move(xi_next) };
  }

  class ExpectedImprovementRandom {};

  template <>
  template <typename Rng, typename BudgetType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
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
    return { std::move(x_next), std::move(xi_next) };
  }

  class ExpectedImprovementKoyama {};

  template <>
  template <typename Rng, typename BudgetType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
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
    return { std::move(x_next), std::move(xi_next) };
  }
}

#endif
