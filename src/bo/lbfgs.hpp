
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

#ifndef __US_GALLERY_LBFGS_HPP__
#define __US_GALLERY_LBFGS_HPP__

#include "../math/blaze.hpp"
#include "../math/mvnormal.hpp"
#include "../system/debug.hpp"
#include "../system/profile.hpp"

#include <nlopt.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace usdg
{
  template <typename Rng,
	    typename ObjectiveFunc>
  inline blaze::DynamicVector<double>
  uniform_random_search(Rng& prng,
			ObjectiveFunc objective_grad,
			size_t n_points,
			size_t n_dims)
  {
    auto x_opt   = blaze::DynamicVector<double>();
    auto x       = blaze::DynamicVector<double>(n_dims);
    double y_opt = std::numeric_limits<double>::lowest();
    auto dist    = std::uniform_real_distribution<double>(0, 1);

    for (size_t i = 0; i < n_points; ++i)
    {
      for (size_t j = 0; j < n_dims; ++j)
      {
	x[j] = dist(prng);
      }
      auto [y, _] = objective_grad(x, false);
      if(y > y_opt)
      {
	x_opt = x;
	y_opt = y;
      }
    }
    return x_opt;
  }

  // template <typename ObjectiveFunc>
  // inline blaze::DynamicVector<double>
  // lbfgs(ObjectiveFunc objective_grad,
  // 	blaze::DynamicVector<double> const& x_init,
  // 	double x_lb,
  // 	double x_ub,
  // 	size_t n_maxeval = 1024,
  // 	double ftol_rel  = 1e-5,
  // 	double xtol_rel  = 1e-5)
  // {
  //   size_t n_dims = x_init.size();
  //   auto x_buf    = blaze::DynamicVector<double>(n_dims);
  //   auto objective_lambda = [&objective_grad, &x_buf](
  //     std::vector<double> const& x,
  //     std::vector<double>& grad) -> double
  //   {
  //     std::copy(x.begin(), x.end(), x_buf.begin());
  //     auto [y, grad_buf] = objective_grad(x_buf, !grad.empty());
  //     std::copy(grad_buf.begin(), grad_buf.end(), grad.begin());
  //     return y;
  //   };

  //   auto objective_wrapped = std::function<
  //     double(std::vector<double> const&,
  // 	     std::vector<double>&)>(objective_lambda);

  //   auto objective_invoke = +[](std::vector<double> const& x,
  // 				std::vector<double>& grad,
  // 				void* punned)
  //   {
  //     return (*reinterpret_cast<
  // 	std::function<
  // 	      double(std::vector<double> const&,
  // 		     std::vector<double>&)>*>(punned))(x, grad);
  //   };
  //   auto optimizer = nlopt::opt(nlopt::LD_LBFGS,
  // 				static_cast<unsigned int>(n_dims));
  //   optimizer.set_lower_bounds(x_lb);
  //   optimizer.set_upper_bounds(x_ub);
  //   optimizer.set_max_objective(objective_invoke, &objective_wrapped);
  //   optimizer.set_xtol_rel(ftol_rel);
  //   optimizer.set_ftol_rel(xtol_rel);
  //   optimizer.set_maxeval(n_maxeval);

  //   try
  //   {
  //     double yval; 
  //     auto result  = optimizer.optimize(x_init, yval);
  //     auto x_found = blaze::DynamicVector<double>(n_dims);
  //     std::copy(x_init.begin(), x_init.end(), x_found.begin());
  //     return x_found;
  //   }
  //   catch(std::exception& e)
  //   {
  //     std::cout << "nlopt failed: " << e.what() << std::endl;
  //   }
  // }

  template <typename Rng, typename ObjectiveFunc>
  inline std::pair<blaze::DynamicVector<double>, double>
  lbfgs_multistage_box_optimize(Rng& prng,
				ObjectiveFunc objective_grad,
				size_t n_dims,
				size_t n_budget,
				size_t n_starts,
				spdlog::logger* logger)
  {
    if(logger)
    {
      logger->info("Optimizing function using multistart L-BFGS: {}",
		   usdg::file_name(__FILE__));
    }

    auto x_opt   = blaze::DynamicVector<double>(); 
    double y_opt = std::numeric_limits<double>::lowest();
    auto x_buf   = blaze::DynamicVector<double>(n_dims);

    auto objective_lambda = [&objective_grad, &x_buf](
      std::vector<double> const& x,
      std::vector<double>& grad) -> double
    {
      std::copy(x.begin(), x.end(), x_buf.begin());
      auto [y, grad_buf] = objective_grad(x_buf, !grad.empty());
      std::copy(grad_buf.begin(), grad_buf.end(), grad.begin());
      return y;
    };
    auto objective_wrapped = std::function<
      double(std::vector<double> const&,
	     std::vector<double>&)>(objective_lambda);

    auto objective_invoke = +[](std::vector<double> const& x,
				std::vector<double>& grad,
				void* punned)
    {
      return (*reinterpret_cast<
	std::function<
	      double(std::vector<double> const&,
		     std::vector<double>&)>*>(punned))(x, grad);
    };

    auto x_init = std::vector<double>(n_dims);
    for (size_t i = 0; i < n_starts; ++i)
    {
      auto best_init = usdg::uniform_random_search(prng, objective_grad, n_budget, n_dims);
      std::copy(best_init.begin(), best_init.end(), x_init.begin());
      auto optimizer = nlopt::opt(nlopt::LD_LBFGS,
				  static_cast<unsigned int>(n_dims));
      optimizer.set_lower_bounds(0.0);
      optimizer.set_upper_bounds(1.0);
      optimizer.set_max_objective(objective_invoke, &objective_wrapped);
      optimizer.set_xtol_rel(1e-4);
      optimizer.set_ftol_rel(1e-5);
      optimizer.set_maxeval(128);

      try
      {
	double yval; 
	auto result  = optimizer.optimize(x_init, yval);
	auto x_found = blaze::DynamicVector<double>(n_dims);
	std::copy(x_init.begin(), x_init.end(), x_found.begin());

	auto [y, _] = objective_grad(x_found, false);
	if (y > y_opt)
	{
	  x_opt = x_found;
	  y_opt = y;
	}
      }
      catch(std::exception& e)
      {
	std::cout << "nlopt failed: " << e.what() << std::endl;
      }
    }

    if(logger)
    {
      logger->info("Optimized function using multistart L-BFGS.");
    }
    return {x_opt, y_opt};
  }

  template <typename Rng, typename ObjectiveFunc>
  inline std::pair<blaze::DynamicVector<double>, double>
  lbfgs_optimize(Rng& prng,
		 ObjectiveFunc objective_grad,
		 size_t n_dims,
		 size_t n_budget,
		 spdlog::logger* logger)
  {
    if(logger)
    {
      logger->info("Optimizing function using L-BFGS: {}",
		   usdg::file_name(__FILE__));
    }

    auto x_opt   = blaze::DynamicVector<double>(); 
    double y_opt = std::numeric_limits<double>::lowest();
    auto x_buf   = blaze::DynamicVector<double>(n_dims);

    auto objective_lambda = [&objective_grad, &x_buf](
      std::vector<double> const& x,
      std::vector<double>& grad) -> double
    {
      std::copy(x.begin(), x.end(), x_buf.begin());
      auto [y, grad_buf] = objective_grad(x_buf, !grad.empty());
      std::copy(grad_buf.begin(), grad_buf.end(), grad.begin());
      return y;
    };
    auto objective_wrapped = std::function<
      double(std::vector<double> const&,
	     std::vector<double>&)>(objective_lambda);

    auto objective_invoke = +[](std::vector<double> const& x,
				std::vector<double>& grad,
				void* punned)
    {
      return (*reinterpret_cast<
	std::function<
	      double(std::vector<double> const&,
		     std::vector<double>&)>*>(punned))(x, grad);
    };

    auto x_init    = usdg::rmvnormal(prng, n_dims);
    auto x_opt_buf = std::vector<double>(n_dims);
    std::copy(x_init.begin(), x_init.end(), x_opt_buf.begin());

    auto optimizer = nlopt::opt(nlopt::LD_LBFGS,
				static_cast<unsigned int>(n_dims));
    optimizer.set_max_objective(objective_invoke, &objective_wrapped);
    optimizer.set_xtol_rel(1e-4);
    optimizer.set_ftol_rel(1e-5);
    optimizer.set_maxeval(128);

    double y_opt_buf; 
    auto result  = optimizer.optimize(x_opt_buf, y_opt_buf);
    auto x_found = blaze::DynamicVector<double>(n_dims);
    std::copy(x_opt_buf.begin(), x_opt_buf.end(), x_found.begin());

    if(logger)
    {
      logger->info("Optimized function using L-BFGS.");
    }
    return {x_found, y_opt_buf};
  }
}

#endif
