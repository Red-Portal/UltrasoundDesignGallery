
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

  double D     = std::numeric_limits<double>::lowest();
  double N     = static_cast<double>(std::distance(begin, end));
  size_t i     = 1;
  for (auto val : values)
  {
    double F       = cdf(val);
    double D_max   = (static_cast<double>(i)/N - cdf(val));
    double D_minus = (F - (static_cast<double>(i)-1)/N); 
    D = std::max(std::max(D_max, D_minus), D);
    ++i;
  }

  double D_thres = 0;
  if(alpha == 0.1 && N == 512)
  {
    D_thres = 0.07582597540719833;
  }
  else if(alpha == 0.05 && N == 512)
  {
    D_thres = 0.08420182387958335;
  }
  else if(alpha == 0.01 && N == 512)
  {
    D_thres = 0.10100517595844977;
  }
  else if(alpha == 0.001 && N == 512)
  {
    D_thres = 0.12104537977466515;
  }
  else if(alpha == 0.0001 && N == 512)
  {
    D_thres = 0.13819077194988733;
  }
  else
  {
    throw std::invalid_argument("invalid KS setting");
  }
  return D > D_thres;
}

inline double
normal_cdf(double x)
{
    return std::erfc(-x/std::sqrt(2))/2;
}

#endif
