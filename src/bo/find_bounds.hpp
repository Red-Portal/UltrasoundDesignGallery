
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

#ifndef __US_GALLERY_FINDBOUNDS_HPP__
#define __US_GALLERY_FINDBOUNDS_HPP__

#include "../math/blaze.hpp"

#include <limits>
#include <cmath>

namespace usdg
{
  inline std::pair<double, double>
  pbo_find_bounds(blaze::DynamicVector<double> const& x,
		  blaze::DynamicVector<double> const& xi)
  {
    double lb     = std::numeric_limits<double>::lowest();
    double ub     = std::numeric_limits<double>::max();
    size_t n_dims = x.size();

    for (size_t i = 0; i < n_dims; ++i)
    {
      if(std::abs(xi[i]) < 1e-5)
      {
	/* no change in this direction */
	continue;
      }

      double alpha = (1 - x[i]) / xi[i];
      if(alpha > 0)
      {
	ub = std::min(alpha, ub);
      }
      else
      {
	lb = std::max(alpha, lb);
      }

      alpha = -x[i] / xi[i];
      if(alpha > 0)
      {
	ub = std::min(alpha, ub);
      }
      else
      {
	lb = std::max(alpha, lb);
      }
    }
    return {lb, ub};
  }

  inline std::tuple<double, double, size_t, double, size_t, double>
  dbounds_dxi(blaze::DynamicVector<double> const& x,
	      blaze::DynamicVector<double> const& xi)
  {
    double lb          = std::numeric_limits<double>::lowest();
    double ub          = std::numeric_limits<double>::max();
    size_t n_dims      = x.size();
    double lb_grad_val = 0.0;
    size_t lb_grad_idx = 0.0;
    double ub_grad_val = 0.0;
    size_t ub_grad_idx = 0.0;

    for (size_t i = 0; i < n_dims; ++i)
    {
      if(std::abs(xi[i]) < 1e-5)
      {
	/* no change in this direction */
	continue;
      }

      double alpha = (1 - x[i]) / xi[i];
      if(alpha > 0)
      {
	if(ub > alpha)
	{
	  ub          = alpha;
	  ub_grad_val = alpha / -xi[i];
	  ub_grad_idx = i;
	}
      }
      else
      {
	if(lb < alpha)
	{
	  lb          = alpha;
	  lb_grad_val = alpha / -xi[i];
	  lb_grad_idx = i;
	}
      }

      alpha = -x[i] / xi[i];
      if(alpha > 0)
      {
	if(ub > alpha)
	{
	  ub          = alpha;
	  ub_grad_val = alpha / -xi[i];
	  ub_grad_idx = i;
	}
      }
      else
      {
	if(lb < alpha)
	{
	  lb          = alpha;
	  lb_grad_val = alpha / -xi[i];
	  lb_grad_idx = i;
	}
      }
    }
    return {lb, ub, lb_grad_idx, lb_grad_val, ub_grad_idx, ub_grad_val};
  }
}

#endif
