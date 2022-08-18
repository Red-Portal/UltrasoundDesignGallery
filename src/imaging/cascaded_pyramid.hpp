
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

#ifndef __US_GALLERY_CASCADED_HPP__
#define __US_GALLERY_CASCADED_HPP__

#include "local_laplacian_pyramid.hpp"
#include "coherent_diffusion.hpp"
#include "complex_diffusion.hpp"
#include "complex_shock.hpp"

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>

namespace usdg
{
  class CascadedPyramid
  {
  private: 
    usdg::FastLocalLaplacianPyramid _pyramid;

    usdg::ComplexShock      _cshock;
    usdg::CoherentDiffusion _ncd1;
    usdg::CoherentDiffusion _ncd2;
    usdg::ComplexDiffusion  _rpncd;

    cv::cuda::GpuMat _img_in_buf;
    cv::cuda::GpuMat _img_out_buf;
    cv::cuda::GpuMat _mask_buf;

  public: 
    CascadedPyramid(size_t n_rows, size_t n_cols);

    void apply(cv::cuda::GpuMat const& image,
	       cv::cuda::GpuMat const& mask,
	       cv::cuda::GpuMat&       output,
	       float llf1_sigma, 
	       float llf2_sigma, 
	       float llf3_sigma, 
	       float cshock_a, 
	       float ncd1_alpha,
	       float ncd1_s,
	       float ncd2_alpha,
	       float ncd2_s,
	       float rpncd_k);

    void apply(cv::Mat const& image,
	       cv::Mat const& mask,
	       cv::Mat&       output,
	       float llf1_sigma, 
	       float llf2_sigma, 
	       float llf3_sigma, 
	       float cshock_a, 
	       float ncd1_alpha,
	       float ncd1_s,
	       float ncd2_alpha,
	       float ncd2_s,
	       float rpncd_k);
  };
}

#endif
