
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
	      cv::Mat& mask,
	      Float in_max_val,
	      Float out_max_val,
	      Float dynamic_range);

  template <>
  inline void
  logcompress<float>(cv::Mat& img,
		     cv::Mat& mask,
		     float in_max_val,
		     float out_max_val,
		     float dynamic_range)
  {
    float in_min_val = powf(10.0f, -dynamic_range / 20.0f)*in_max_val;
    float in_range   = log10f(in_max_val / in_min_val);
    float coeff      = out_max_val / in_range;
    for (int i = 0; i < img.rows; ++i) {
      for (int j = 0; j < img.cols; ++j) {
	if (mask.at<uchar>(i,j) > 0)
	{
	  float x = img.at<float>(i, j);
	  if (x >= in_min_val)
	    img.at<float>(i,j) = coeff * log10f(x / in_min_val);
	  else
	    img.at<float>(i,j) = 0.0f;
	}
	else
	{
	  img.at<float>(i,j) = 0.0f;
	}
      }
    }
  }

  inline void
  logcompress(cv::Mat& img,
	      cv::Mat& mask,
	      double in_max_val,
	      double out_max_val,
	      double dynamic_range)
  {
    if (img.type() == CV_32F)
    {
      logcompress<float>(img,
			 mask,
			 static_cast<float>(in_max_val),
			 static_cast<float>(out_max_val),
			 static_cast<float>(dynamic_range));
    }
    else
    {
      throw std::runtime_error("Only 32-bit floating point is supported!");
    }
  }
}

#endif
