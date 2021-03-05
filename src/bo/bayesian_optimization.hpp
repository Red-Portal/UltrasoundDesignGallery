
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

#ifndef __US_GALLERY_BAYESOPT_HPP__
#define __US_GALLERY_BAYESOPT_HPP__

#include "../gp/data.hpp"
#include "../misc/blaze.hpp"
#include "../misc/debug.hpp"

namespace usdg
{ 
  template <typename AcqFunc>
  class BayesianOptimization
  {
    usdg::Dataset _data;
    blaze::DynamicVector<double> _x_opt;
    double _y_opt;

    inline
    BayesianOptimization(size_t n_dims, size_t n_pseudo);
    
    template <typename Rng>
    inline std::tuple<blaze::DynamicVector<double>,
		      blaze::DynamicVector<double>>
    next_query(Rng& prng,
	       size_t n_dims,
	       size_t n_burn,
	       size_t n_samples,
	       size_t budget) const;

    inline void
    push_data(blaze::DynamicVector<double> const& x,
	      blaze::DynamicVector<double> const& xi,
	      double alpha);
  };

  template <typename AcqFunc>
  inline
  BayesianOptimization<AcqFunc>::
  BayesianOptimization(size_t n_dims, size_t n_pseudo)
    : _data(n_dims, n_pseudo),
      _x_opt(),
      _y_opt(std::numeric_limits<double>::lowest()) 
  { }

  template <typename AcqFunc>
  inline void
  BayesianOptimization<AcqFunc>::
  push_data(blaze::DynamicVector<double> const& x,
	    blaze::DynamicVector<double> const& xi,
	    double alpha)
  {
    auto betas = blaze::DynamicVector<double>();
    auto point = usdg::Datapoint{alpha, betas, xi, x};
    this->_data.push_back(std::move(point));
  }
}

#endif
