

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

#include <optional>

namespace usdg
{
  class OptimizationManager
  {
  private:
    blaze::DynamicVector<double> _x;
    blaze::DynamicVector<double> _xi;
    double _beta_lb;
    double _beta_ub;

    usdg::BayesianOptimization<usdg::EI_Koyama> _optimizer;
  public:
    
    blaze::DynamicVector<double> query(double beta);

    blaze::DynamicVector<double> best(double beta);

    void reset();
  };
}

#endif
