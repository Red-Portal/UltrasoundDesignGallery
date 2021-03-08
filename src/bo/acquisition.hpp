
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
#include "../gp/marginalized_gp.hpp"
#include "../misc/blaze.hpp"
#include "../misc/debug.hpp"
#include "../misc/linearalgebra.hpp"
#include "../misc/mvnormal.hpp"
#include "bayesian_optimization.hpp"
#include "cmaes.hpp"
#include "sample_hyper.hpp"

#include <vector>
#include <random>
#include <limits>

namespace usdg
{
  class ThompsonSamplingKoyama {};
  class ThompsonSampling {};
  class ExpectedImprovementKoyama {};
  class ExpectedImprovement {};
// class ExpectedImprovement {};

  template <typename KernelType>
  inline std::pair<blaze::DynamicVector<double>, double>
  find_best_alpha(usdg::Dataset const& data,
		  blaze::DynamicMatrix<double> const& data_mat,
		  usdg::MarginalizedGP<KernelType> const& mgp)
  {
    double y_opt = std::numeric_limits<double>::lowest(); 
    auto x_opt   = blaze::DynamicVector<double>();
    for (size_t i = 0; i < data.num_data(); ++i)
    {
      auto x_alpha     = blaze::column(data_mat, data.alpha_index(i));
      auto [mean, var] = mgp.predict(data_mat, x_alpha);
      if (mean > y_opt)
      {
	y_opt = mean;
	x_opt = x_alpha;
      }
    }
    return { std::move(x_opt), y_opt };
  }

  template <>
  template <typename Rng>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
  BayesianOptimization<usdg::ThompsonSamplingKoyama>::
  next_query(Rng& prng,
	     size_t n_burn,
	     size_t n_samples,
	     size_t budget,
	     //blaze::DynamicVector<double> const& theta_init,
	     usdg::MvNormal<usdg::DiagonalChol> const& prior_dist,
	     spdlog::logger* logger) const
  {
    if(logger)
    {
      logger->info("Finding next Bayesian optimization query with Thomson sampling with the Koyama scheme: {}",
		   usdg::file_name(__FILE__));
    }

    auto data_mat = this->_data.data_matrix();
    auto [theta_samples, f_samples, K_samples] = usdg::sample_gp_hyper(
      prng,
      this->_data,
      data_mat,
      n_burn,
      n_samples,
      prior_dist.mean,
      prior_dist,
      logger);

    auto mgp = usdg::MarginalizedGP<usdg::Matern52Iso>(blaze::exp(theta_samples),
						       f_samples,
						       K_samples);
    auto dist   = std::uniform_int_distribution<size_t>(0, n_samples-1);
    auto i      = dist(prng);
    auto theta  = blaze::column(theta_samples, i);
    auto f      = blaze::column(f_samples, i);
    auto K      = K_samples[i];
    auto kernel = usdg::Matern52Iso(blaze::exp(theta));
    auto alpha  = usdg::solve(K, f);
    auto gp     = LatentGaussianProcess{
      std::move(K), std::move(alpha), std::move(kernel)};

    auto ts_acq = [&](blaze::DynamicVector<double> const& x) {
      auto [mean, var] = gp.predict(data_mat, x);
      return -mean;
    };
    auto [x_champ, y_champ] = usdg::cmaes_optimize(
      prng, ts_acq, this->_n_dims, budget, logger);


    auto [x_opt, y_opt] = usdg::find_best_alpha(this->_data, data_mat, mgp);
    auto delta = (x_opt - x_champ);
    auto xi    = blaze::evaluate(delta / blaze::max(blaze::abs(delta)));

    if(logger)
    {
      logger->info("Found next Bayesian optimization query.");
    }
    return {std::move(x_champ), std::move(xi)};
  }

  template <>
  template <typename Rng>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
  BayesianOptimization<usdg::ThompsonSampling>::
  next_query(Rng& prng,
	     size_t n_burn,
	     size_t n_samples,
	     size_t budget,
	     usdg::MvNormal<usdg::DiagonalChol> const& prior_dist,
	     spdlog::logger* logger) const
  {
    if(logger)
    {
      logger->info("Finding next Bayesian optimization query with Thomson sampling: {}",
		   usdg::file_name(__FILE__));
    }

    auto data_mat = this->_data.data_matrix();
    auto [theta_samples, f_samples, K_samples] = usdg::sample_gp_hyper(
      prng,
      this->_data,
      data_mat,
      n_burn,
      n_samples,
      prior_dist.mean,
      prior_dist,
      logger);

    auto dist   = std::uniform_int_distribution<size_t>(0, n_samples-1);
    auto idx    = dist(prng);
    auto theta  = blaze::column(theta_samples, idx);
    auto f      = blaze::column(f_samples, idx);
    auto K      = K_samples[idx];
    auto kernel = usdg::Matern52Iso(blaze::exp(theta));
    auto alpha  = usdg::solve(K, f);
    auto gp     = LatentGaussianProcess{
      std::move(K), std::move(alpha), std::move(kernel)};

    auto ts_x_acq = [&](blaze::DynamicVector<double> const& x) {
      auto [mean, var] = gp.predict(data_mat, x);
      return -mean;
    };
    auto [x_champ, y_x_champ] = usdg::cmaes_optimize(
      prng, ts_x_acq, this->_n_dims, budget/2, logger);

    auto ts_xi_acq = [&](blaze::DynamicVector<double> const& xi) {
      size_t n_beta   = 8;
      auto [lb, ub]   = usdg::pbo_find_bounds(x_champ, xi);
      auto beta_delta = (ub - lb)/static_cast<double>(n_beta);
      auto y_avg      = 0.0;
      for (size_t i = 0; i < n_beta; ++i)
      {
	auto beta        = lb + beta_delta*static_cast<double>(i);
	auto [mean, var] = gp.predict(data_mat, x_champ + beta*xi);
	y_avg += mean;
      }
      return -y_avg/static_cast<double>(n_beta);
    };
    auto [xi_champ, y_xi_champ] = usdg::cmaes_maxball_optimize(
      prng, ts_xi_acq, this->_n_dims, budget/2, logger);

    if(logger)
    {
      logger->info("Found next Bayesian optimization query.");
    }
    return {std::move(x_champ), std::move(xi_champ)};
  }

  template <>
  template <typename Rng>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
  BayesianOptimization<usdg::ExpectedImprovement>::
  next_query(Rng& prng,
	     size_t n_burn,
	     size_t n_samples,
	     size_t budget,
	     usdg::MvNormal<usdg::DiagonalChol> const& prior_dist,
	     spdlog::logger* logger) const
  {
    if(logger)
    {
      logger->info("Finding next Bayesian optimization query with expected improvement: {}",
		   usdg::file_name(__FILE__));
    }

    auto data_mat = this->_data.data_matrix();
    auto [theta_samples, f_samples, K_samples] = usdg::sample_gp_hyper(
      prng,
      this->_data,
      data_mat,
      n_burn,
      n_samples,
      prior_dist.mean,
      prior_dist,
      nullptr);

    auto mgp = usdg::MarginalizedGP<usdg::Matern52Iso>(blaze::exp(theta_samples),
						       f_samples,
						       K_samples);

    auto [_, y_opt] = usdg::find_best_alpha(this->_data, data_mat, mgp);
    size_t n_pseudo = this->_data.num_pseudo();
    size_t n_dims   = this->_n_dims;
    auto ei_x_acq = [&](blaze::DynamicVector<double> const& x_in) {
      auto [mean, var] = mgp.predict(data_mat, x_in);
      double sigma     = sqrt(var);
      double delta     = (y_opt - mean);
      double ei        = delta*usdg::normal_cdf(delta/sigma)
	+ sigma*usdg::dnormal(delta/sigma);
      return ei;
    };
    auto [x_champ, y_x_champ] = cmaes_optimize(
      prng, ei_x_acq, n_dims, budget/2, logger);

    auto ei_xi_acq = [&](blaze::DynamicVector<double> const& xi_in) {
      size_t n_pseudo    = 8;
      auto [lb, ub]      = usdg::pbo_find_bounds(x_champ, xi_in);
      double beta_delta  = (ub - lb) / static_cast<double>(n_pseudo);
      double ei_avg      = 0.0;
      for (size_t i = 0; i < n_pseudo; ++i)
      {
	auto beta        = lb + beta_delta*static_cast<double>(i);
	auto [mean, var] = mgp.predict(data_mat, x_champ + beta*xi_in);
	double sigma     = sqrt(var);
	double delta     = (y_opt - mean);
	double ei        = delta*usdg::normal_cdf(delta/sigma)
	  + sigma*usdg::dnormal(delta/sigma);
	ei_avg += ei;
      }
      return -ei_avg / static_cast<double>(n_pseudo);
    };
    auto [xi_champ, y_xi_champ] = cmaes_optimize(
      prng, ei_xi_acq, n_dims, budget/2, logger);

    if(logger)
    {
      logger->info("Found next Bayesian optimization query.");
    }
    return { std::move(x_champ), std::move(xi_champ) };
  }

  template <>
  template <typename Rng>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
  BayesianOptimization<usdg::ExpectedImprovementKoyama>::
  next_query(Rng& prng,
	     size_t n_burn,
	     size_t n_samples,
	     size_t budget,
	     usdg::MvNormal<usdg::DiagonalChol> const& prior_dist,
	     spdlog::logger* logger) const
  {
    if(logger)
    {
      logger->info("Finding next Bayesian optimization query with expected improvement and Koyama scheme: {}",
		   usdg::file_name(__FILE__));
    }

    auto data_mat = this->_data.data_matrix();
    auto [theta_samples, f_samples, K_samples] = usdg::sample_gp_hyper(
      prng,
      this->_data,
      data_mat,
      n_burn,
      n_samples,
      prior_dist.mean,
      prior_dist,
      nullptr);

    auto mgp = usdg::MarginalizedGP<usdg::Matern52Iso>(blaze::exp(theta_samples),
						       f_samples,
						       K_samples);
    auto [x_opt, y_opt] = usdg::find_best_alpha(this->_data, data_mat, mgp);

    size_t n_pseudo = this->_data.num_pseudo();
    size_t n_dims   = this->_n_dims;
    auto ei_x_acq = [&](blaze::DynamicVector<double> const& x_in) {
      auto [mean, var] = mgp.predict(data_mat, x_in);
      double sigma     = sqrt(var);
      double delta     = (y_opt - mean);
      double ei        = delta*usdg::normal_cdf(delta/sigma)
	+ sigma*usdg::dnormal(delta/sigma);
      return ei;
    };
    auto [x_champ, y_x_champ] = cmaes_optimize(
      prng, ei_x_acq, n_dims, budget, logger);

    auto delta = x_opt - x_champ;
    auto xi    = delta / blaze::max(blaze::abs(delta));
    if(logger)
    {
      logger->info("Found next Bayesian optimization query.");
    }
    return { std::move(x_champ), std::move(xi) };
  }

  // class ExpectedImprovement {};

  // template <>
  // template <typename Rng>
  // inline std::tuple<blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>>
  // BayesianOptimization<usdg::ExpectedImprovement>::
  // next_query(Rng& prng,
  // 	     size_t n_burn,
  // 	     size_t n_samples,
  // 	     size_t budget,
  // 	     usdg::MvNormal<usdg::DiagonalChol> const& prior_dist,
  // 	     spdlog::logger* logger) const
  // {
  //   if(logger)
  //   {
  //     logger->info("Finding next Bayesian optimization query with Thomson sampling: {}",
  // 		   usdg::file_name(__FILE__));
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

  //   auto mgp = usdg::MarginalizedGP<usdg::Matern52Iso>(blaze::exp(theta_samples),
  // 						       f_samples,
  // 						       K_samples);
  //   auto [_, y_opt] = usdg::find_best_alpha(this->_data, data_mat, mgp);

  //   size_t n_pseudo = this->_data.num_pseudo();
  //   size_t n_dims   = this->_n_dims;
  //   auto ei_acq = [&](blaze::DynamicVector<double> const& x_in) {
  //     size_t K = 16;
  //     auto x        = blaze::subvector(x_in, 0,      n_dims);
  //     auto xi_unorm = blaze::subvector(x_in, n_dims, n_dims);
  //     auto xi       = blaze::evaluate(xi_unorm / blaze::max(blaze::abs(xi_unorm)));

  //     auto [lb, ub]  = usdg::pbo_find_bounds(x, xi);
  //     auto beta_dist = std::uniform_real_distribution<double>(lb, ub);
  //     double ei_sum  = 0.0;
  //     for (size_t i = 0; i < K; ++i)
  //     {
  // 	double z = std::numeric_limits<double>::lowest();
  // 	for (size_t j = 0; j < n_pseudo; ++j)
  // 	{
  // 	  auto beta   = beta_dist(prng);
  // 	  auto x_beta = x + beta*xi;

  // 	  auto [mean, var] = mgp.predict(data_mat, x_beta);
  // 	  auto pred_dist   = std::normal_distribution<double>(mean, sqrt(var));
  // 	  double y         = pred_dist(prng);
  // 	  if (y > z)
  // 	  {
  // 	    z = y;
  // 	  }
  // 	}
  // 	ei_sum += std::max(z - y_opt, 0.0);
  //     }
  //     return -ei_sum / static_cast<double>(K);
  //   };
  //   auto [champ_x, champ_y] = cmaes_optimize(
  //     prng, ei_acq, n_dims*2, budget, logger);

  //   auto x  = blaze::subvector(champ_x, 0,      n_dims);
  //   auto xi = blaze::subvector(champ_x, n_dims, n_dims);

  //   // if(logger)
  //   // {
  //   //   log->info("Found next Bayesian optimization query: {}",
  //   // 		usdg::file_name(__FILE__));
  //   // }

  //   return {x, xi};
  // }
}

#endif
