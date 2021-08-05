
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

#ifndef __US_GALLERY_TSONAD_HPP__
#define __US_GALLERY_TSONAD_HPP__

#include <cstddef>

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudafilters.hpp>

namespace usdg
{
  class IEEDShock
  {
  private:
    cv::cuda::GpuMat _mask;
    cv::cuda::GpuMat _img_buf1;
    cv::cuda::GpuMat _img_buf2;

    cv::cuda::GpuMat _G_x;
    cv::cuda::GpuMat _G_y;

    cv::cuda::GpuMat _J_xx;
    cv::cuda::GpuMat _J_xy;
    cv::cuda::GpuMat _J_yy;

    cv::cuda::GpuMat _J_xx_rho;
    cv::cuda::GpuMat _J_xy_rho;
    cv::cuda::GpuMat _J_yy_rho;

    cv::cuda::GpuMat _D_xx;
    cv::cuda::GpuMat _D_xy;
    cv::cuda::GpuMat _D_yy;

    cv::cuda::GpuMat _edge_map;
    cv::cuda::GpuMat _shock;

    cv::Ptr<cv::cuda::Filter> _gaussian_filter;

  public:
    IEEDShock();

    void preallocate(size_t n_rows, size_t n_cols);

    void apply(cv::Mat const& image,
	       cv::Mat const& mask,
	       cv::Mat&       output,
	       float m1, float m2,
	       float k1, float k2,
	       float Cm1, float Cm2,
	       float r,
	       float dt, int n_iters);
  };
}

#endif
