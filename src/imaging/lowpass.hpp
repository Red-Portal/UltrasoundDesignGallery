
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

#ifndef __US_GALLERY_BUTTERWORTH_HPP__
#define __US_GALLERY_BUTTERWORTH_HPP__

#include <opencv4/opencv2/core/core.hpp>

#include <cstddef>

namespace usdg
{
  class ButterworthLPF
  {
  private:
    cv::Size _fft_size;
    cv::Mat _image_padded;
    cv::Mat _image_spectrum;
    cv::Mat _kernel_spectrum;
    cv::Mat _filtered_spectrum;
    cv::Mat _filtered;
    
  public:
    ButterworthLPF();

    void preallocate(size_t M, size_t N);

    void apply(cv::Mat const& img,
	       cv::Mat& out,
	       float cutoff_theta,
	       float order);
  };
}

#endif
