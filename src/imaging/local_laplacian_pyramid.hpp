
/*
 * Copyright (C) 2022 Kyurae Kim
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

#ifndef __US_GALLERY_LOCAL_LAPLACIAN_PYRAMID_HPP__
#define __US_GALLERY_LOCAL_LAPLACIAN_PYRAMID_HPP__

#include "gaussian_pyramid.hpp"
#include "laplacian_pyramid.hpp"

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <vector>
#include <stdexcept>

#include <cmath>
#include <cstddef>

namespace usdg
{
  class FastLocalLaplacianPyramid
  {
  private:
    usdg::LaplacianPyramid _L;

    size_t _n_quants;
    usdg::GaussianPyramid  _G;
    std::vector<cv::cuda::GpuMat> _masks;
    std::vector<usdg::LaplacianPyramid> _L_quants;

    cv::cuda::GpuMat _remap_buffer;
    cv::cuda::GpuMat _img_buffer;
    cv::cuda::GpuMat _mask_buffer;

    void
    remap_image(cv::cuda::GpuMat const& img,
		cv::cuda::GpuMat const& mask,
		float g,
		float alpha,
		float beta,
		float sigma_range,
		float I_range,
		cv::cuda::GpuMat& L_out) const;

    void
    interpolate_laplacian_pyramids(std::vector<usdg::LaplacianPyramid>& L_quants,
				   cv::cuda::GpuMat const& G,
				   cv::cuda::GpuMat const& mask,
				   size_t level,
				   float I_range,
				   cv::cuda::GpuMat& L_out) const;

  public:
    FastLocalLaplacianPyramid(size_t n_scales, size_t n_fourier);

    void preallocate(size_t n_rows, size_t n_cols);

    void apply(cv::cuda::GpuMat const& img,
	       cv::cuda::GpuMat const& mask,
	       float alpha,
	       float beta,
	       float sigma_range);

    void apply(cv::Mat const& img,
	       cv::Mat const& mask,
	       float alpha,
	       float beta,
	       float sigma_range);

    inline cv::cuda::GpuMat&
    L(size_t idx)
    {
      return _L.L(idx);
    }

    inline cv::cuda::GpuMat&
    mask(size_t idx)
    {
      return _L_quants[0].mask(idx);
    }
  };
}

#endif
