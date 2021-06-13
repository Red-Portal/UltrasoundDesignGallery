
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

#ifndef __US_GALLERY_PMADSHOCK_HPP__
#define __US_GALLERY_PMADSHOCK_HPP__

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudafilters.hpp>

namespace usdg
{
  class PMADShock
  {
  private:
    cv::cuda::GpuMat _G_buf1;
    cv::cuda::GpuMat _G_buf2;
    cv::cuda::GpuMat _L_buf1;
    cv::cuda::GpuMat _L_buf2;
    cv::cuda::GpuMat _eta_buf1;
    cv::cuda::GpuMat _eta_buf2;
    cv::Ptr<cv::cuda::Filter> _eta_filter;

  public:
    PMADShock();

    void preallocate(size_t n_rows, size_t n_cols);

    void apply(cv::Mat const& G,
	       cv::Mat const& L,
	       cv::Mat&       output,
	       float dt,
	       float r,
	       float K,
	       size_t niters);
  };
}

#endif
