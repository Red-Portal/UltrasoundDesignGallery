
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
      if (std::abs(xi[i]) < 1e-5)
      {
	/* no change in this direction */
	continue;
      }

      if (xi[i] > 0)
      {
	double alpha_ub = (1 - x[i]) / xi[i];
	double alpha_lb = -x[i]      / xi[i];
	lb = std::max(alpha_lb, lb);
	ub = std::min(alpha_ub, ub);
      }
      else
      {
	double alpha_lb = (1 - x[i]) / xi[i];
	double alpha_ub = -x[i]      / xi[i];
	lb = std::max(alpha_lb, lb);
	ub = std::min(alpha_ub, ub);
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

      if(xi[i] > 0)
      {
	double alpha_ub = (1 - x[i]) / xi[i];
	double alpha_lb = -x[i]      / xi[i];
	if(ub > alpha_ub)
	{
	  ub          = alpha_ub;
	  ub_grad_val = alpha_ub / -xi[i];
	  ub_grad_idx = i;
	}
	if(lb < alpha_lb)
	{
	  lb          = alpha_lb;
	  lb_grad_val = alpha_lb / -xi[i];
	  lb_grad_idx = i;
	}
      }
      else
      {
	double alpha_lb = (1 - x[i]) / xi[i];
	double alpha_ub = -x[i]      / xi[i];
	if(ub > alpha_ub)
	{
	  ub          = alpha_ub;
	  ub_grad_val = alpha_ub / -xi[i];
	  ub_grad_idx = i;
	}
	if(lb < alpha_lb)
	{
	  lb          = alpha_lb;
	  lb_grad_val = alpha_lb / -xi[i];
	  lb_grad_idx = i;
	}
      }
    }
    return {lb, ub, lb_grad_idx, lb_grad_val, ub_grad_idx, ub_grad_val};
  }
}

#endif
