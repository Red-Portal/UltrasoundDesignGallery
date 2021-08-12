
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

#include <nlohmann/json.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

#include <chrono>
#include <ctime>

namespace usdg
{
  size_t n_beta   = 20;
  size_t n_init   = 4;
  size_t n_budget = 10000;

  OptimizationManager::
  OptimizationManager()
    : _prng(1u),
      _lock(),
      _n_dims(custom_ip_dimension()),
      _x(),
      _xi(),
      _x_opt(),
      _thread(),
      _iteration(0),
      _is_optimizing(false),
      _presets(),
      _optimizer(_n_dims, n_beta)
  {
    _x   = usdg::rmvuniform(_prng, _n_dims, 0, 1);
    _xi  = usdg::rmvnormal( _prng, _n_dims);
    _xi /= blaze::max(blaze::abs(_xi)); 
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

    if(_iteration < n_init)
    {
      if (_presets.size() > _iteration)
      {
	_lock.lock();
	_x  = _presets[_iteration].first;
	_xi = _presets[_iteration].second;
	_lock.unlock();
      }
      else
      {
	auto x   = usdg::rmvuniform(_prng, _n_dims, 0, 1);
	auto xi  = usdg::rmvnormal( _prng, _n_dims);
	xi      /= blaze::max(blaze::abs(xi)); 

	_lock.lock();
	_x     = std::move(x);
	_xi    = std::move(xi);
	_lock.unlock();
      }
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

  std::string
  OptimizationManager::
  serialize()
  {
    auto& data      = _optimizer._data;
    size_t n_points = data.num_data();
    auto json       = nlohmann::json();
    auto timepoint  = std::chrono::system_clock::now();
    auto ctimepoint = std::chrono::system_clock::to_time_t(timepoint);
    auto datetime   = std::ctime(&ctimepoint);
    json["date"]    = datetime;
    for (size_t i = 0; i < n_points; ++i)
    {
      auto json_datapoint = nlohmann::json();
      auto x_vec          = std::vector<double>(data.dims());
      auto xi_vec         = std::vector<double>(data.dims());
      auto beta_vec       = std::vector<double>(data.num_pseudo());

      std::copy(data[i].x.cbegin(),     data[i].x.cend(),     x_vec.begin());
      std::copy(data[i].xi.cbegin(),    data[i].xi.cend(),    xi_vec.begin());
      std::copy(data[i].betas.cbegin(), data[i].betas.cend(), beta_vec.begin());
      json_datapoint["x"]     = std::move(x_vec);
      json_datapoint["xi"]    = std::move(xi_vec);
      json_datapoint["alpha"] = data[i].alpha;
      json_datapoint["beta"]  = std::move(beta_vec);
      json["data"].push_back(std::move(json_datapoint));
    }
    return json.dump(2);
  }

  void
  OptimizationManager::
  deserialize_impl(std::string const& json_data)
  {
    auto& data      = _optimizer._data;
    auto parsed     = nlohmann::json::parse(json_data)["data"];
    size_t n_points = parsed.size();

    if (n_points < 2)
    {
      return;
    }
     
    for (size_t i = 0; i < n_points; ++i)
    {
      auto& datapoint = parsed[i];
      auto& x         = datapoint["x"];
      auto& xi        = datapoint["xi"];
      auto& beta      = datapoint["beta"];
      double alpha    = datapoint["alpha"];

      auto x_blaze    = blaze::DynamicVector<double>(x.size());
      auto xi_blaze   = blaze::DynamicVector<double>(xi.size());
      auto beta_blaze = blaze::DynamicVector<double>(beta.size());

      std::copy(x.begin(),    x.end(),    x_blaze.begin());
      std::copy(xi.begin(),   xi.end(),   xi_blaze.begin());
      std::copy(beta.begin(), beta.end(), beta_blaze.begin());

      if (i == n_points-1)
      {
	_x  = std::move(x_blaze);
	_xi = std::move(xi_blaze);
	this->find_next_query(alpha);
      }
      else
      {
	data.push_back(
	  usdg::Datapoint(alpha,
			  std::move(beta_blaze),
			  std::move(xi_blaze),
			  std::move(x_blaze)));
      }
    }
    _iteration = n_points;
  }

  void
  OptimizationManager::
  load_preset(std::string const& json_data)
  {
    auto parsed      = nlohmann::json::parse(json_data);
    size_t n_presets = parsed.size();

    _lock.lock();
    for (size_t i = 0; i < n_presets; ++i)
    {
      auto x  = parsed[i]["x"];
      auto xi = parsed[i]["xi"];

      auto x_blaze  = blaze::DynamicVector<double>(x.size());
      auto xi_blaze = blaze::DynamicVector<double>(xi.size());
      std::copy(x.begin(),  x.end(),  x_blaze.begin());
      std::copy(xi.begin(), xi.end(), xi_blaze.begin());
      _presets.emplace_back(std::move(x_blaze), std::move(xi_blaze));
    }
    _x  = _presets[0].first;
    _xi = _presets[0].second;
    _lock.unlock();
  }

  void
  OptimizationManager::
  deserialize(std::string const& json_data)
  {
    this->deserialize_impl(json_data);
  }

  bool
  OptimizationManager::
  is_optimizing()
  {
    return _is_optimizing.load();
  }

  size_t
  OptimizationManager::
  iteration()
  {
    return _iteration;
  }
}

