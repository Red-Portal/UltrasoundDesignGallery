
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

#include "local_laplacian_pyramid.hpp"
//#include "cuda_utils.hpp"

#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numbers>

#include <cmath>

namespace usdg
{
  GaussianLocalLaplacianPyramid::
  GaussianLocalLaplacianPyramid(size_t n_levels,
				size_t n_fourier,
				float decimation_ratio)
    : _decimation_ratio(decimation_ratio),
      _n_fourier(n_fourier),
      _masks(n_levels),
      _L(n_levels),
      _img_buffer(),
      _mask_buffer(),
      _G_up_buffer(),
      _I_cos(),
      _I_sin(),
      _G_cos_up(),
      _G_sin_up(),
      _G(n_levels, decimation_ratio),
      _G_cos(n_levels, decimation_ratio),
      _G_sin(n_levels, decimation_ratio)
  {
    if (decimation_ratio <= 1)
      throw std::runtime_error("Decimation ratio should be larger than 1");
  }

  void
  GaussianLocalLaplacianPyramid::
  preallocate(size_t n_rows, size_t n_cols)
  {
    size_t n_levels  = _G.levels();

    _img_buffer.create(static_cast<int>(n_rows),
		       static_cast<int>(n_cols),
		       CV_32F);
    _mask_buffer.create(static_cast<int>(n_rows),
			static_cast<int>(n_cols),
			CV_8U);

    _G_up_buffer.create(static_cast<int>(n_rows),
			static_cast<int>(n_cols),
			CV_32F);
    _I_cos.create(static_cast<int>(n_rows),
		  static_cast<int>(n_cols),
		  CV_32F);
    _I_sin.create(static_cast<int>(n_rows),
		  static_cast<int>(n_cols),
		  CV_32F);
    _G_cos_up.create(static_cast<int>(n_rows),
		     static_cast<int>(n_cols),
		     CV_32F);
    _G_sin_up.create(static_cast<int>(n_rows),
		     static_cast<int>(n_cols),
		     CV_32F);

    _G.preallocate(    n_rows, n_cols);
    _G_cos.preallocate(n_rows, n_cols);
    _G_sin.preallocate(n_rows, n_cols);

    for (size_t i = 0; i < n_levels; ++i)
    {
      _L[i].create(static_cast<int>(n_rows),
		   static_cast<int>(n_cols),
		   CV_32F);
    }
  }

  void
  GaussianLocalLaplacianPyramid::
  apply(cv::cuda::GpuMat const& image,
	cv::cuda::GpuMat const& mask,
	//std::vector<float> const& sigma_range,
	float sigma)
  {
    size_t n_levels  = _G.levels();

    int T       = 260;
    auto pi     = std::numbers::pi;

    _G.apply(image, sigma);

    /* Construct masks and Laplacian pyramid */
    mask.copyTo(_masks[0]);
    for (size_t l = 0; l < n_levels-1; ++l)
    {
      auto curr_size = _G.G(l).size();
      cv::cuda::resize(mask,      _masks[l],    curr_size);
      cv::cuda::resize(_G.G(l+1), _G_up_buffer, curr_size);
      cv::cuda::subtract(_G.G(l), _G_up_buffer, _L[l]);
    }
    _L[n_levels-1] = _G.G(n_levels-1);

    /* Construct local Laplacian residual */
    for (size_t k = 0; k < _n_fourier; ++k)
    {
      float m           = -1.0;
      float omega       = 2*pi*k/T;
      float sigma_r     = 30.0;
      float sigma_r2    = sigma_r*sigma_r;
      float omega2      = omega*omega;
      float alpha       = sigma_r*sqrtf(2*pi)/T*expf(sigma_r2*omega2/-2.f);
      float alpha_tilde = 2*alpha*omega*sigma_r2;

      this->compute_fourier_series(image, mask, omega, T, _I_cos, _I_sin);
      _G_cos.apply(_I_cos, sigma);
      _G_sin.apply(_I_sin, sigma);

      for (size_t l = 0; l < n_levels-1; ++l)
      {
	auto& G_cos_curr = _G_cos.G(l);
	auto& G_sin_curr = _G_sin.G(l);
	auto& G_cos_next = _G_cos.G(l+1);
	auto& G_sin_next = _G_sin.G(l+1);
	auto size_curr   = G_cos_curr.size();
	cv::cuda::resize(G_cos_next, _G_cos_up, size_curr);
	cv::cuda::resize(G_sin_next, _G_sin_up, size_curr);

	if (l == 0)
	  this->fourier_firstlayer_accumulate(alpha_tilde,
					      omega,
					      m,
					      _G.G(l),
					      _G_cos_up,
					      _G_sin_up,
					      _masks[l],
					      _L[l]);
	else
	  this->fourier_recon_accumulate(alpha_tilde,
					 omega,
					 m,
					 _G.G(l),
					 G_cos_curr,
					 G_sin_curr,
					 _G_cos_up,
					 _G_sin_up,
					 _masks[l],
					 _L[l]);
      }
    }
  }

  void
  GaussianLocalLaplacianPyramid::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	//std::vector<float> const& sigma_range,
	float sigma)
  {
    _img_buffer.upload(image);
    _mask_buffer.upload(mask);
    this->apply(_img_buffer, _mask_buffer, sigma);
  }
}
