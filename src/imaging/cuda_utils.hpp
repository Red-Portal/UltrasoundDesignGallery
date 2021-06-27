
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

#ifndef __US_GALLERY_CUDAUTILS_HPP__
#define __US_GALLERY_CUDAUTILS_HPP__

namespace usdg
{
  __device__ __forceinline__ float
  tukey_biweight(float x, float sigma)
  {
    if (abs(x) < sigma)
    {
      float ratio = x / sigma;
      float coef  = 1.0 - ratio*ratio;
      return coef*coef;
    }
    else
    {
      return 0.0f;
    }
  }

  __device__ __forceinline__ int sign(float x)
  { 
    int t = x < 0 ? -1 : 0;
    return x > 0 ? 1 : t;
  }

  __device__ __forceinline__ float minmod(float x, float y)
  { 
    if(x*y > 0)
    {
      return sign(x)*min(abs(x), abs(y));
    }
    else
    {
      return 0.0;
    }
  }

}

#endif
