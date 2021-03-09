
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

#ifndef __US_GALLERY_CMAES_HPP__
#define __US_GALLERY_CMAES_HPP__

#include "../math/blaze.hpp"
#include "../math/debug.hpp"

#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include <vector>
#include <functional>

namespace usdg
{
  struct BoundedProblem {
    std::function<double(blaze::DynamicVector<double> const&)> acq;
    std::vector<double> lb;
    std::vector<double> ub;

    inline std::vector<double>
    fitness(std::vector<double> const& x) const
    {
      auto x_in = blaze::DynamicVector<double>(x.size(), x.data());
      return { acq(x_in) };
    }

    inline std::pair<std::vector<double>,
		     std::vector<double>>
    get_bounds() const
    {
      return { lb, ub };
    }
  };

  struct BallConstrainedProblem {
    std::function<double(blaze::DynamicVector<double> const&)> acq;
    std::vector<double> lb;
    std::vector<double> ub;

    inline std::vector<double>
    fitness(std::vector<double> const& x) const
    {
      auto x_in     = blaze::DynamicVector<double>(x.size(), x.data());
      auto f        = acq(x_in);
      auto max_norm = blaze::max(blaze::abs(x_in));
      return { f, max_norm };
    }

    inline std::pair<std::vector<double>,
		     std::vector<double>>
    get_bounds() const
    {
      return { lb, ub };
    }

    inline pagmo::vector_double::size_type
    get_nec() const
    {
      return 1;
    }

    inline pagmo::vector_double::size_type
    get_nic() const
    {
      return 0;
    }
  };

  template <typename Rng,
	    typename ObjectiveFunc>
  inline std::pair<blaze::DynamicVector<double>, double>
  cmaes_optimize(Rng& prng,
		 ObjectiveFunc objective,
		 size_t n_dims,
		 size_t budget,
		 spdlog::logger* logger)
  {
    if(logger)
    {
      logger->info("Optimizing function using CMA-ES: {}",
		   usdg::file_name(__FILE__));
    }
    double sigma0  = sqrt(static_cast<double>(n_dims))/4;
    double ftol    = 1e-6;
    double xtol    = 1e-3;
    size_t n_pop   = 4 + static_cast<size_t>(
      ceil(3*log(static_cast<double>(n_dims))));
    auto unif_dist = std::uniform_real_distribution<double>(0, 1);
    auto prob      = pagmo::problem(
      usdg::BoundedProblem{
	objective,
	std::vector<double>(n_dims, 0.0),
	std::vector<double>(n_dims, 1.0)}
      );
    auto pop = pagmo::population{prob};
    for (size_t i = 0; i < n_pop; ++i)
    {
      auto vec = std::vector<double>(n_dims);
      for (size_t j = 0; j < n_dims; ++j)
      {
	vec[j] = unif_dist(prng);
      }
      pop.push_back(std::move(vec));
    }

    unsigned int budg_tmp = static_cast<unsigned int>(budget);
    auto user_algo = pagmo::cmaes{budg_tmp, -1, -1, -1, -1,
      sigma0, ftol, xtol, false, true};
    user_algo.set_verbosity(1u);
    pop = user_algo.evolve(pop);

    auto champ_x = pop.champion_x();
    auto champ_f = pop.champion_f()[0];
    if(logger)
    {
      logger->info("Optimized function using CMA-ES.");
    }
    return {blaze::DynamicVector<double>(champ_x.size(), champ_x.data()), champ_f};
  }

  template <typename Rng,
	    typename ObjectiveFunc>
  inline std::pair<blaze::DynamicVector<double>, double>
  cmaes_maxball_optimize(Rng& prng,
			 ObjectiveFunc objective,
			 size_t n_dims,
			 size_t budget,
			 spdlog::logger* logger)
  {
    if(logger)
    {
      logger->info("Optimizing function with max-norm ball constraint using CMA-ES: {}",
		   usdg::file_name(__FILE__));
    }
    double sigma0  = sqrt(static_cast<double>(n_dims))/4;
    double ftol    = 1e-6;
    double xtol    = 1e-3;
    size_t n_pop   = 4 + static_cast<size_t>(
      ceil(3*log(static_cast<double>(n_dims))));
    auto norm_dist = std::normal_distribution<double>(0, 1);
    auto prob      = pagmo::problem(
      usdg::BallConstrainedProblem{
	objective,
	std::vector<double>(n_dims, -1.0),
	std::vector<double>(n_dims,  1.0)}
      );
    prob.set_c_tol({1e-3});
    auto pop = pagmo::population{prob};
    for (size_t i = 0; i < n_pop; ++i)
    {
      auto vec = std::vector<double>(n_dims);
      for (size_t j = 0; j < n_dims; ++j)
      {
	vec[j] = norm_dist(prng);
      }
      /* Apply max-norm constraint */
      auto vec_max = *std::max_element(vec.begin(), vec.end());
      for (size_t j = 0; j < n_dims; ++j)
      {
	vec[j] /= vec_max;
      }
      pop.push_back(std::move(vec));
    }

    unsigned int budg_tmp = static_cast<unsigned int>(budget);
    auto inner_algo = pagmo::cmaes{budg_tmp/16, -1, -1, -1, -1,
      sigma0, ftol, xtol, false, true};
    auto user_algo  = pagmo::cstrs_self_adaptive{16, std::move(inner_algo)};
    user_algo.set_verbosity(1u);
    pop = user_algo.evolve(pop);

    auto champ_x     = pop.champion_x();
    auto champ_f     = pop.champion_f()[0];
    auto champ_x_vec = blaze::DynamicVector<double>(champ_x.size(), champ_x.data());
    champ_x_vec     /= blaze::max(blaze::abs(champ_x_vec));

    if(logger)
    {
      logger->info("Optimized function with max-norm ball constraint using CMA-ES.");
    }
    return {std::move(champ_x_vec), champ_f};
  }
}

#endif
