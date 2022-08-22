
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

#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <stdexcept>
#include <algorithm>

#include <cmath>

namespace usdg
{
  FastLocalLaplacianPyramid::
  FastLocalLaplacianPyramid(size_t n_levels, size_t n_quants)
    : _L(n_levels),
      _n_quants(n_quants),
      _G(n_levels),
      _masks(n_levels),
      _L_quants(),
      _remap_buffer(),
      _img_buffer(),
      _mask_buffer()
  {
    _L_quants.reserve(n_quants);
    for (size_t n = 0; n < n_quants; ++n)
    {
      _L_quants.emplace_back(n_levels);
    }
  }

  void
  FastLocalLaplacianPyramid::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _remap_buffer.create(static_cast<int>(n_rows),
			 static_cast<int>(n_cols),
			 CV_32F);
    _img_buffer.create(static_cast<int>(n_rows),
		       static_cast<int>(n_cols),
		       CV_32F);
    _mask_buffer.create(static_cast<int>(n_rows),
			static_cast<int>(n_cols),
			CV_8U);

    _G.preallocate(n_rows, n_cols);
    _L.preallocate(n_rows, n_cols);

    // auto imag_init = cv::cuda::GpuMat(n_rows, n_cols, CV_32F, cv::Scalar(0));
    // auto mask_init = cv::cuda::GpuMat(n_rows, n_cols, CV_8U,  cv::Scalar(1));
    // _G.apply(imag_init, 2.0, 2.0);
    // _L.apply(imag_init, mask_init, 2.0, 2.0);

    for (size_t n = 0; n < _n_quants; ++n)
    {
      _L_quants[n].preallocate(n_rows, n_cols);
    }

    for (size_t l = 0; l < _G.levels(); ++l)
    {
      _masks[l].create(static_cast<int>(n_rows),
		       static_cast<int>(n_cols),
		       CV_8U);
    }
  }

  void
  FastLocalLaplacianPyramid::
  apply(cv::cuda::GpuMat const& image,
	cv::cuda::GpuMat const& mask,
	float alpha,
	float beta,
	float sigma_range)
  {
    size_t n_levels = _G.levels();
    double I_min    = 0.;
    double I_max    = 255.;
    double I_range  = I_max - I_min;

    _G.apply(image, 2.0, 2.0);

    for (size_t l = 0; l < n_levels; ++l)
    {
      cv::cuda::resize(mask, _masks[l], _G.G(l).size(), cv::INTER_LINEAR);
    }

    for (size_t n = 0; n < _n_quants; ++n)
    {
      auto start = std::chrono::steady_clock::now();
      float g = I_min + n*(I_range/(_n_quants - 1));
      this->remap_image(image,
			mask,
			g,
			alpha,
			beta,
			sigma_range,
			I_range,
			_remap_buffer);
      _L_quants[n].apply(_remap_buffer, mask, 2.0, 2.0);
    }

    for (size_t l = 0; l < n_levels - 1; ++l)
    {
      _L.L(l).create(_G.G(l).rows, _G.G(l).cols, CV_32F);
      this->interpolate_laplacian_pyramids(_L_quants,
					   _G.G(l),
					   _masks[l],
					   l,
					   I_range,
					   _L.L(l));
    }
    auto& G_last = _G.G(n_levels - 1);
    auto& L_last = _L.L(n_levels - 1);
    G_last.copyTo(L_last);
  }

  void
  FastLocalLaplacianPyramid::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	float alpha,
	float beta,
	float sigma_range)
  {
    _img_buffer.upload(image);
    _mask_buffer.upload(mask);
    this->apply(_img_buffer, _mask_buffer, alpha, beta, sigma_range);
  }
}
