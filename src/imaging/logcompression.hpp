
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

#ifndef __US_GALLERY_LOGCOMPRESSION_HPP__
#define __US_GALLERY_LOGCOMPRESSION_HPP__

#include <cmath>

namespace usdg
{
  template <typename Float>
  inline void
  logcompress(cv::Mat& img,
	      Float in_max_val,
	      Float out_max_val,
	      Float dynamic_range)
  {
    Float in_min_val = pow(10.0, -dynamic_range / 20.0)*in_max_val;
    Float in_range   = log10(in_max_val / in_min_val);
    Float coeff      = out_max_val / in_range;
    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
	Float x = img.at<Float>(i, j);
	if (x >= in_min_val)
	  img.at<Float>(i,j) = coeff * log10(x / in_min_val);
	else
	  img.at<Float>(i,j) = 0.0;
      }
    }
  }

  inline void
  logcompress(cv::Mat& img,
	      double in_max_val,
	      double out_max_val,
	      double dynamic_range)
  {
    if (img.type() == CV_32F)
      logcompress<float>(img, in_max_val, out_max_val, dynamic_range);
    else if (img.type() == CV_64F)
      logcompress<double>(img, in_max_val, out_max_val, dynamic_range);
  }
}

#endif
