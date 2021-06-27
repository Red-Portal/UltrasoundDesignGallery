
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

#ifndef __US_GALLERY_OSRAD_HPP__
#define __US_GALLERY_OSRAD_HPP__

#include <cstddef>

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudafilters.hpp>

namespace usdg
{
  class OSRAD
  {
  private:
    cv::cuda::GpuMat _G_buf1;
    cv::cuda::GpuMat _G_buf2;
    cv::cuda::GpuMat _L_buf1;
    cv::cuda::GpuMat _L_buf2;

    cv::cuda::GpuMat _Dxx_buf;
    cv::cuda::GpuMat _Dxy_buf;
    cv::cuda::GpuMat _Dyy_buf;

    cv::Ptr<cv::cuda::Filter> _gaussian_filter;

  public:
    OSRAD();

    void preallocate(size_t n_rows, size_t n_cols);

    void apply(cv::Mat const& G_image,
	       cv::Mat const& L_image,
	       cv::Mat&       output,
	       float dt,
	       float sigma_r,
	       float sigma_g,
	       float ctang,
	       size_t niters);
  };
}

#endif
