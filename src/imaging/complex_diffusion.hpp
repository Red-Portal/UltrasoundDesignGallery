
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

#ifndef __US_GALLERY_RPNCD_HPP__
#define __US_GALLERY_RPNCD_HPP__

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <cmath>

namespace usdg
{
  class ComplexDiffusion
  {
    /* 
     * Ramp-Preserving nonlinear Complex Diffusion
     * 
     * "Image enhancement and denoising by complex diffusion processes"
     * Gilboa, Guy, et al.,
     * IEEE Transactions on Pattern Analysis and Machine Intelligence, 2004
     */
  private:
    cv::cuda::GpuMat _img_buf1;
    cv::cuda::GpuMat _img_buf2;
    cv::cuda::GpuMat _coeff;

    cv::cuda::GpuMat _img_in_buf;
    cv::cuda::GpuMat _img_out_buf;
    cv::cuda::GpuMat _mask_buf;

  public:
    ComplexDiffusion();

    void preallocate(size_t n_rows, size_t n_cols);

    void apply(cv::cuda::GpuMat const& image,
	       cv::cuda::GpuMat const& mask,
	       cv::cuda::GpuMat&       output,
	       float k,  float theta, float dt, int n_iters);

    void apply(cv::Mat const& image,
	       cv::Mat const& mask,
	       cv::Mat&       output,
	       float k,  float theta, float dt, int n_iters);
  };
}

#endif
