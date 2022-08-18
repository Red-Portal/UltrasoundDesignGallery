
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

#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <stdexcept>
#include <iostream>

#include <cmath>

namespace usdg
{
  LaplacianPyramid::
  LaplacianPyramid(size_t n_scales)
    : _L(n_scales),
      _masks(n_scales),
      _G_l(),
      _G_l_next(),
      _G_l_next_up()
  { }

  void
  LaplacianPyramid::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _G_l.create(        n_rows, n_cols, CV_32F);
    _G_l_next.create(   n_rows, n_cols, CV_32F);
    _G_l_next_up.create(n_rows, n_cols, CV_32F);

    auto n_scales = _L.size();
    for (size_t l = 0; l < n_scales; ++l)
    {
      _L[l].create(    n_rows, n_cols, CV_32F);
      _masks[l].create(n_rows, n_cols, CV_8U);
    }
  }

  void
  LaplacianPyramid::
  apply(cv::cuda::GpuMat const& image,
	cv::cuda::GpuMat const& mask,
	float decimation_ratio,
	float sigma)
  {
    (void)decimation_ratio;
    (void)sigma;
    (void)mask;
    //if (decimation_ratio <= 1)
    //  throw std::runtime_error("Decimation ratio should be larger than 1");

    size_t n_levels = _L.size();

    image.copyTo(_G_l);
    for (size_t l = 0; l < n_levels-1; ++l)
    {
      cv::cuda::pyrDown(_G_l, _G_l_next);
      cv::cuda::resize(_G_l_next, _G_l_next_up, _G_l.size());
      cv::cuda::resize(mask,      _masks[l],    _G_l.size());
      cv::cuda::subtract(_G_l, _G_l_next_up,  _L[l], _masks[l]);
      cv::swap(_G_l_next, _G_l);
    }
    cv::swap(_G_l_next, _G_l);
    _G_l_next.copyTo(_L.back());
    cv::cuda::resize(mask, _masks.back(), _L.back().size());
  }

  // void
  // LaplacianPyramid::
  // apply(cv::Mat const& image,
  // 	float decimation_ratio,
  // 	float sigma)
  // {
  //   _img_buffer.upload(image);
  //   this->apply(_img_buffer, decimation_ratio, sigma);
  // }
}
