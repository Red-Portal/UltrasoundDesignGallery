
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

#include "../misc/blaze.hpp"

#include <limits>

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
}

#endif
