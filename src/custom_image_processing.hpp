
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

#ifndef _CUSTOM_IMAGE_PROCESSING_HPP_
#define _CUSTOM_IMAGE_PROCESSING_HPP_

#include <opencv4/opencv2/core/utility.hpp>

#include "math/blaze.hpp"
#include "imaging/lpndsf.hpp"

namespace usdg
{
  struct CustomImageProcessing
  {
    usdg::LPNDSF _process;

    CustomImageProcessing(size_t n_rows,
			  size_t n_cols)
      : _process(n_rows, n_cols)
    { }
    
    inline void
    apply(cv::Mat const& input,
	  cv::Mat& output,
	  blaze::DynamicVector<double> const& param)
    {
      (void)param;
      float r0 = 0.0f;
      float r1 = 0.1f;
      float r2 = 0.1f;

      float k0 = 0.1f;
      float k1 = 0.1f;
      float k2 = 0.01f;
      float k3 = 0.01f;

      float t0 = 2.0f;
      float t1 = 5.0f;
      float t2 = 2.0f;
      float t3 = 2.0f;

      float alpha = 0.8f;
      float beta  = 1.2f;

      _process.apply(input, output,
		     r0, k0, t0,
		     r1, k1, t1,
		     r2, k2, t2,
		     k3, t3,
		     alpha, beta);
    }
  };
}

#endif
