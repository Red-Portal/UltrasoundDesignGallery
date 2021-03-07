
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

#include "../misc/blaze.hpp"
#include "../misc/debug.hpp"
#include "../misc/linearalgebra.hpp"
#include "bayesian_optimization.hpp"
#include "cmaes.hpp"
#include "sample_hyper.hpp"

#include <vector>
#include <random>

namespace usdg
{
  class ThompsonSampling {};

  template <>
  template <typename Rng>
  inline std::pair<blaze::DynamicVector<double>,
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
      prior_dist,
      nullptr);

    auto dist   = std::uniform_int_distribution<size_t>(0, n_samples);
    auto i      = dist(prng);
    auto theta  = blaze::column(theta_samples, i);
    auto f      = blaze::column(f_samples, i);
    auto K      = K_samples[i];
    auto kernel = usdg::Matern52Iso(theta);
    auto alpha  = usdg::solve(K, f);
    auto gp     = LatentGaussianProcess{
      std::move(K), std::move(alpha), std::move(kernel)};

    auto ts_acq = [&](blaze::DynamicVector<double> const& x) {
      auto [mean, var] = gp.predict(data_mat, x);
      return -mean;
    };
    auto [champ_x, champ_y] = cmaes_optimize(
      prng, ts_acq, this->_n_dims, budget, logger);

    auto last_dp = this->_data._data.back();
    auto prev_x  = (last_dp.xi * last_dp.alpha) + last_dp.x;
    auto delta   = (prev_x - champ_x);
    auto xi      = blaze::evaluate(delta / blaze::max(blaze::abs(delta)));

    // if(logger)
    // {
    //   log->info("Found next Bayesian optimization query: {}",
    // 		usdg::file_name(__FILE__));
    // }

    return {std::move(champ_x), std::move(xi)};
  }

  // class ExpectedImprovement {};

  // template <>
  // template <typename Rng>
  // inline std::tuple<blaze::DynamicVector<double>,
  // 		    blaze::DynamicVector<double>>
  // BayesianOptimization<usdg::ExpectedImprovement>::
  // next_query(Rng& prng,
  // 	     size_t n_dims,
  // 	     size_t n_burn,
  // 	     size_t n_samples,
  // 	     size_t budget,
  // 	     spdlog::logger* logger) const
  // {
  //   if(logger)
  //   {
  //     log->info("Finding next Bayesian optimization query with expected improvement: {}",
  // 		usdg::file_name(__FILE__));
  //   }

  //   auto data_mat = this->_data.data_matrix();
  //   auto [theta_samples, f_samples, K_samples] = sample_gp_hyper(
  //     prng, this->_data, data_mat, n_dims, n_burn, n_samples);

  //   auto ei_acq = [](){};
  //   cmaes_optimize(prng, ei_acq, n_dims, budget, logger)
  // }
}

#endif
