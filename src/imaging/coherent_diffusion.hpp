
/*
 * Copyright (C) 2021-2022 Kyurae Kim
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

#ifndef __US_GALLERY_NCD_HPP__
#define __US_GALLERY_NCD_HPP__

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudafilters.hpp>

#include <cmath>

namespace usdg
{
  class CoherentDiffusion
  {
    /* 
     * Non-linear Coherent Diffusion
     * 
     *  Abd-Elmoniem, Khaled Z. et al.
     * "Real-time speckle reduction and coherence enhancement 
     *  in ultrasound imaging via nonlinear anisotropic diffusion"
     *  IEEE Transactions on Biomedical Engineering, 2002
     */
  private:
    cv::cuda::GpuMat _img_buf1;
    cv::cuda::GpuMat _img_buf2;

    cv::cuda::GpuMat _J_xx;
    cv::cuda::GpuMat _J_xy;
    cv::cuda::GpuMat _J_yy;

    cv::cuda::GpuMat _J_xx_rho;
    cv::cuda::GpuMat _J_xy_rho;
    cv::cuda::GpuMat _J_yy_rho;

    cv::cuda::GpuMat _D_xx;
    cv::cuda::GpuMat _D_xy;
    cv::cuda::GpuMat _D_yy;

    cv::cuda::GpuMat _G_x;
    cv::cuda::GpuMat _G_y;

    cv::cuda::GpuMat _j1;
    cv::cuda::GpuMat _j2;

    cv::cuda::GpuMat _img_in_buf;
    cv::cuda::GpuMat _img_out_buf;
    cv::cuda::GpuMat _mask_buf;

  public:
    CoherentDiffusion();

    void preallocate(size_t n_rows, size_t n_cols);

    void apply(cv::cuda::GpuMat const& image,
	       cv::cuda::GpuMat const& mask,
	       cv::cuda::GpuMat&       output,
	       float rho,  float alpha, float s, float dt, int n_iters);

    void apply(cv::Mat  const& image,
	       cv::Mat  const& mask,
	       cv::Mat&        output,
	       float rho,  float alpha, float s, float dt, int n_iters);
  };
}

#endif
