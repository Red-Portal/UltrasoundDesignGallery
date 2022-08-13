
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

#include "gaussian_pyramid.hpp"

//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <stdexcept>

#include <cmath>

namespace usdg
{
  GaussianPyramid::
  GaussianPyramid(size_t n_levels,
		  float decimation_ratio)
    : _G(n_levels),
      _decimation_ratio(decimation_ratio),
      _img_buffer()
  {
    if (decimation_ratio <= 1)
      throw std::runtime_error("Decimation ratio should be larger than 1");
  }

  void
  GaussianPyramid::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _G[0].create(n_rows, n_cols, CV_32F);
    for (size_t i = 1; i < _G.size(); ++i)
    {
      _G[i].create(n_rows, n_cols, CV_32F);
    }
  }

  void
  GaussianPyramid::
  apply(cv::cuda::GpuMat const& image, float sigma)
  {
    size_t n_rows = image.rows;
    size_t n_cols = image.cols;

    image.copyTo(_G[0]);
    for (size_t l = 1; l < _G.size(); ++l)
    {
      cv::cuda::pyrDown(_G[l-1], _G[l]);
    }
  }

  void
  GaussianPyramid::
  apply(cv::Mat const& image, float sigma)
  {
    _img_buffer.upload(image);
    this->apply(_img_buffer, sigma);
  }
}
