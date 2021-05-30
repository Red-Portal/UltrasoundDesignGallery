
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
#include "../system/debug.hpp"
#include "../system/profile.hpp"

#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include <chrono>
#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace usdg
{
  template <class, template <class, class...> class>
  struct is_instance : public std::false_type {};

  template <class...Ts, template <class, class...> class U>
  struct is_instance<U<Ts...>, U> : public std::true_type {};

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
      auto x_in = blaze::DynamicVector<double>(x.size(), x.data());
      x_in     /= blaze::max(blaze::abs(x_in));
      auto f    = acq(x_in);
      return { f };
    }

    inline std::pair<std::vector<double>,
		     std::vector<double>>
    get_bounds() const
    {
      return { lb, ub };
    }
  };

  template <typename RatioType>
  inline bool
  terminate(std::chrono::duration<double, RatioType> budget,
	    usdg::clock::time_point const& start_time,
	    size_t n_evals)
  {
    return (usdg::clock::now() - start_time) > budget;
  }

  template <typename IntType,
	    typename = std::enable_if_t<std::is_integral<IntType>::value>>
  inline bool
  terminate(IntType budget,
	    usdg::clock::time_point const& start_time,
	    size_t n_evals)
  {
    (void)start_time;
    return n_evals > budget;
  }

  template <typename Rng,
	    typename ObjectiveFunc,
	    typename BudgetType>
  inline std::pair<blaze::DynamicVector<double>, double>
  cmaes_optimize(Rng& prng,
		 ObjectiveFunc objective,
		 size_t n_dims,
		 BudgetType budget,
		 spdlog::logger* logger)
  {
    if(logger)
    {
      logger->info("Optimizing function using CMA-ES: {}",
		   usdg::file_name(__FILE__));
    }
    double sigma0 = sqrt(static_cast<double>(n_dims))/4;
    double ftol   = 1e-6;
    double xtol   = 1e-3;
    size_t n_pop  = 4 + static_cast<size_t>(
      ceil(3*log(static_cast<double>(n_dims))));

    size_t n_feval    = 0;
    auto start_time  = usdg::clock::now();
    auto unif_dist   = std::uniform_real_distribution<double>(0, 1);
    auto prob        = pagmo::problem(
      usdg::BoundedProblem{
	[&](blaze::DynamicVector<double> const& x)
	{
	  n_feval += 1;
	  return objective(x);
	},
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

    auto seed      = static_cast<unsigned int>(prng());
    auto user_algo = pagmo::cmaes{1, -1, -1, -1, -1,
      sigma0, ftol, xtol, true, true, seed};
    //user_algo.set_verbosity(1u);

    while(true)
    {
      size_t before_n_feval  = n_feval;
      pop = user_algo.evolve(pop);
      size_t after_n_feval  = n_feval;
      if(usdg::terminate(budget, start_time, n_feval)
	 || after_n_feval - before_n_feval == 0)
      {
	break;
      }
    }

    auto champ_x = pop.champion_x();
    auto champ_f = pop.champion_f()[0];
    if(logger)
    {
      logger->info("Optimized function using CMA-ES.");
    }
    auto champ_x_blaze = blaze::DynamicVector<double>(champ_x.size());
    std::copy(champ_x_blaze.begin(), champ_x_blaze.end(), champ_x_blaze.begin());
    return {champ_x_blaze, champ_f};
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

    size_t n_feval    = 0;
    auto start_time  = usdg::clock::now();
    auto norm_dist   = std::normal_distribution<double>(0, 1);
    auto prob        = pagmo::problem(
      usdg::BallConstrainedProblem{
	[&](blaze::DynamicVector<double> const& x)
	{
	  n_feval += 1;
	  return objective(x);
	},
	std::vector<double>(n_dims, 0.0),
	std::vector<double>(n_dims, 1.0)}
      );
    prob.set_c_tol({1e-3});
    auto pop = pagmo::population{prob};
    for (size_t i = 0; i < n_pop; ++i)
    {
      auto vec = std::vector<double>(n_dims);
      for (size_t j = 0; j < n_dims; ++j)
      {
	vec[j] = std::abs(norm_dist(prng));
      }
      pop.push_back(std::move(vec));
    }

    auto seed      = static_cast<unsigned int>(prng());
    auto user_algo = pagmo::cmaes{1, -1, -1, -1, -1,
      sigma0, ftol, xtol, true, true, seed};

    while(true)
    {
      size_t before_n_feval = n_feval;
      pop = user_algo.evolve(pop);
      size_t after_n_feval  = n_feval;
      if(usdg::terminate(budget, start_time, n_feval)
	 || after_n_feval - before_n_feval == 0)
      {
	break;
      }
    }

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
