
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

#include "laplacian_pyramid.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#include <stdexcept>
#include <iostream>

#include <cmath>

namespace usdg
{
  LaplacianPyramid::
  LaplacianPyramid(size_t n_scales)
    : _L(n_scales), _G(n_scales), _masks(n_scales), _blur_buffer()
  { }

  void
  LaplacianPyramid::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	float decimation_ratio,
	float sigma)
  {
    if (decimation_ratio <= 1)
      throw std::runtime_error("Decimation ratio should be larger than 1");

    size_t M = image.rows;
    size_t N = image.cols;

    image.copyTo(_G[0]);
    mask.copyTo(_masks[0]);
    for (size_t i = 1; i < _G.size(); ++i)
    {
      size_t M_dec = static_cast<size_t>(ceil(M/pow(decimation_ratio, i)));
      size_t N_dec = static_cast<size_t>(ceil(N/pow(decimation_ratio, i)));
      cv::GaussianBlur(_G[i-1], _blur_buffer, cv::Size(5, 5), sigma, sigma);
      _L[i-1] = _G[i-1] - _blur_buffer;
      cv::resize(_blur_buffer, _G[i],     cv::Size(N_dec, M_dec));
      cv::resize(mask,         _masks[i], cv::Size(N_dec, M_dec));
    }
    _G.back().copyTo(_L.back());
  }
}
