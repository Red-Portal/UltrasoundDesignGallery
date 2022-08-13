
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

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <vector>
#include <stdexcept>

#include <cmath>
#include <cstddef>


namespace usdg
{
  class GaussianLocalLaplacianPyramid
  {
  private:
    float _decimation_ratio;
    size_t _n_fourier;
    std::vector<cv::cuda::GpuMat> _masks;
    std::vector<cv::cuda::GpuMat> _L;

    cv::cuda::GpuMat _img_buffer;
    cv::cuda::GpuMat _mask_buffer;

    cv::cuda::GpuMat _G_up_buffer;
    cv::cuda::GpuMat _I_cos;
    cv::cuda::GpuMat _I_sin;
    cv::cuda::GpuMat _G_cos_up;
    cv::cuda::GpuMat _G_sin_up;

    usdg::GaussianPyramid _G;
    usdg::GaussianPyramid _G_cos;
    usdg::GaussianPyramid _G_sin;

    void
    fourier_firstlayer_accumulate(float alpha_tilde,
				  float omega,
				  float m,
				  cv::cuda::GpuMat const& G,
				  cv::cuda::GpuMat const& G_cos,
				  cv::cuda::GpuMat const& G_sin,
				  cv::cuda::GpuMat const& mask,
				  cv::cuda::GpuMat& L_fourier_recon) const;

    void
    fourier_recon_accumulate(float alpha_tilde,
			     float omega,
			     float m,
			     cv::cuda::GpuMat const& G,
			     cv::cuda::GpuMat const& G_cos,
			     cv::cuda::GpuMat const& G_sin,
			     cv::cuda::GpuMat const& G_cos_up,
			     cv::cuda::GpuMat const& G_sin_up,
			     cv::cuda::GpuMat const& mask,
			     cv::cuda::GpuMat& L_fourier_recon) const;

    void
    compute_fourier_series(cv::cuda::GpuMat const& img_in,
			   cv::cuda::GpuMat const& mask,
			   float omega,
			   int T,
			   cv::cuda::GpuMat& img_cos_out,
			   cv::cuda::GpuMat& img_sin_out) const;

  public:
    GaussianLocalLaplacianPyramid(size_t n_scales,
				  size_t n_fourier,
				  float decimation_ratio);

    void preallocate(size_t n_rows, size_t n_cols);

    void apply(cv::cuda::GpuMat const& img,
	       cv::cuda::GpuMat const& mask,
	       float dec_sigma);

    void apply(cv::Mat const& img,
	       cv::Mat const& mask,
	       float dec_sigma);

    inline cv::cuda::GpuMat&
    G(size_t idx)
    {
      return _G.G(idx);
    }

    inline cv::Mat
    fuck(size_t idx)
    {
      auto n_rows = _L[idx].rows;
      auto n_cols = _L[idx].cols;
      auto out    = cv::Mat();
      out.create(n_rows, n_cols, CV_32F);
      _L[idx].download(out);
      return out;
    }

    inline cv::cuda::GpuMat&
    mask(size_t idx)
    {
      return _masks[idx];
    }
  };
}

#endif
