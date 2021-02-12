
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


#ifndef __US_GALLERY_EMPCDF_HPP__
#define __US_GALLERY_EMPCDF_HPP__

#include <blaze/math/DynamicVector.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

#include <iostream>

template <typename CdfFunc, typename It>
inline bool
kolmogorov_smirnoff_test(double alpha,
			 CdfFunc cdf,
			 It begin,
			 It end)
/* 
 * Returns false if failed to reject hypo that x comes from the distribution of cdf.
 *
 * For a more principled implementation, refer to:
 * Gonzalez, Teofilo, Sartaj Sahni, and William R. Franta. 
 * "An efficient algorithm for the Kolmogorov-Smirnov and Lilliefors tests." 
 * ACM Transactions on Mathematical Software (TOMS) 3.1 (1977): 60-64. 
 */
{
  assert(alpha < 1.0 && alpha > 0.0);

  auto values = std::vector<double>(begin, end);
  std::sort(values.begin(), values.end());

  double D     = std::numeric_limits<double>::min();
  double N     = static_cast<double>(std::distance(begin, end));
  double sqrtN = sqrt(N);
  size_t i     = 1;
  for (auto val : values)
  {
    double F       = cdf(val);
    double D_max   = (static_cast<double>(i)/N - cdf(val));
    double D_minus = (F - (static_cast<double>(i)-1)/N); 
    D = std::max(std::max(D_max, D_minus), D);
    ++i;
  }

  double p_thres = 0;
  if(alpha == 0.1)
  {
    p_thres = 1.22385 / sqrtN;
  }
  else if(alpha == 0.05)
  {
    p_thres = 1.35810 / sqrtN;
  }
  else if(alpha == 0.01)
  {
    p_thres = 1.62762 / sqrtN;
  }
  return D > p_thres;
}

inline double
normal_cdf(double x)
{
    return std::erfc(-x/std::sqrt(2))/2;
}

#endif
