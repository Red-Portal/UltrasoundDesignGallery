
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

#ifndef __US_GALLERY_OPTIMIZATIONMANAGER_HPP__
#define __US_GALLERY_OPTIMIZATIONMANAGER_HPP__

#include "../bo/bayesian_optimization.hpp"
#include "../bo/acquisition.hpp"
#include "../math/blaze.hpp"
#include "../math/prng.hpp"

#include <string>
#include <atomic>
#include <mutex>
#include <optional>
#include <thread>

namespace usdg
{
  using XVector  = blaze::DynamicVector<double>;
  using XiVector = blaze::DynamicVector<double>;

  class OptimizationManager
  {
  private:
    usdg::Random123              _prng;
    std::mutex                   _lock;

    size_t                       _n_dims;
    blaze::DynamicVector<double> _x;
    blaze::DynamicVector<double> _xi;
    blaze::DynamicVector<double> _x_opt;
    std::thread                  _thread;
    size_t                       _iteration;
    std::atomic<bool>            _is_optimizing;

    std::vector<std::pair<XVector, XiVector>> _presets;

    usdg::BayesianOptimization<usdg::EI_Koyama> _optimizer;

    blaze::DynamicVector<double>
    transform_unit_beta(double beta_unit,
			blaze::DynamicVector<double> const& x,
			blaze::DynamicVector<double> const& xi);

    void find_next_query_impl(double beta);

    void deserialize_impl(std::string const& json_data);

  public:
    OptimizationManager();

    void find_next_query(double beta);
    
    blaze::DynamicVector<double> query(double beta);

    blaze::DynamicVector<double> best();

    std::string serialize();

    void deserialize(std::string const& json_data);

    void load_preset(std::string const& json_data);

    bool is_optimizing();

    size_t iteration();
  };
}

#endif
