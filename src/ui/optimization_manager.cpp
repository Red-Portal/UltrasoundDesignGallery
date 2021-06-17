
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

#include "optimization_manager.hpp"
#include "../bo/find_bounds.hpp"
#include "../math/uniform.hpp"
#include "../math/mvnormal.hpp"
#include "../custom_image_processing.hpp"

namespace usdg
{
  size_t n_beta   = 20;
  size_t n_init   = 4;
  size_t n_budget = 1000;

  OptimizationManager::
  OptimizationManager()
    : _prng(0u),
      _lock(),
      _n_dims(custom_ip_dimension()),
      _x(),
      _xi(),
      _x_opt(),
      _thread(),
      _iteration(0),
      _is_optimizing(false),
      _optimizer(_n_dims, n_beta)
  {
    _x     = usdg::rmvuniform(_prng, _n_dims, 0, 1);
    _xi    = usdg::rmvnormal( _prng, _n_dims);
    _xi   /= blaze::max(blaze::abs(_xi)); 
    _x_opt = _x;
  }

  blaze::DynamicVector<double>
  OptimizationManager::
  transform_unit_beta(double beta_unit,
		      blaze::DynamicVector<double> const& x,
		      blaze::DynamicVector<double> const& xi)
  {
    auto [beta_lb, beta_ub] = usdg::pbo_find_bounds(x, xi);
    double beta = beta_unit * (beta_ub - beta_lb) + beta_lb;
    return x + xi*beta;
  }

  blaze::DynamicVector<double>
  OptimizationManager::
  query(double beta_unit)
  {
    auto lock = std::scoped_lock(_lock);
    return this->transform_unit_beta(beta_unit, _x, _xi);
  }

  blaze::DynamicVector<double>
  OptimizationManager::
  best()
  {
    auto lock = std::scoped_lock(_lock);
    return _x_opt;
  }

  void
  OptimizationManager::
  find_next_query_impl(double alpha_unit)
  {
    double eps = 1e-5;
    auto [beta_lb, beta_ub] = usdg::pbo_find_bounds(_x, _xi);
    double alpha = alpha_unit * (beta_ub - beta_lb) + beta_lb;
    auto betas   = blaze::DynamicVector<double>(n_beta);
    blaze::subvector(betas, 2, n_beta-2) =  usdg::sample_beta(_prng,
							      alpha,
							      beta_lb + eps,
							      beta_ub - eps,
							      0,
							      n_beta-2,
							      _n_dims);
    betas[0] = beta_lb;
    betas[1] = beta_ub;
    _optimizer.push_data(_x, _xi, betas, alpha);

    if(_iteration < 4)
    {
      auto x   = usdg::rmvuniform(_prng, _n_dims, 0, 1);
      auto xi  = usdg::rmvnormal( _prng, _n_dims);
      xi      /= blaze::max(blaze::abs(xi)); 

      _lock.lock();
      _x     = std::move(x);
      _xi    = std::move(xi);
      _x_opt = std::move(x);
      _lock.unlock();
    }
    else
    {
      auto [x, xi, x_opt, _] = _optimizer.next_query(_prng,
						     _iteration,
						     n_beta,
						     n_budget,
						     custom_ip_scale(),
						     nullptr,
						     nullptr);
      _lock.lock();
      _x     = std::move(x);
      _xi    = std::move(xi);
      _x_opt = std::move(x_opt);
      _lock.unlock();
    }
    _iteration += 1;
  }

  void
  OptimizationManager::
  find_next_query(double alpha_unit)
  {
    _thread = std::thread([this, alpha_unit]{
      _is_optimizing.store(true);
      this->find_next_query_impl(alpha_unit);
      _is_optimizing.store(false);
    });
    _thread.detach();
  }

  bool
  OptimizationManager::
  is_optimizing()
  {
    return _is_optimizing.load();
  }
}

