
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
#include <opencv4/opencv2/cudafilters.hpp>

#include <stdexcept>

#include <cmath>

namespace usdg
{
  GaussianPyramid::
  GaussianPyramid(size_t n_levels)
    : _G(n_levels),
      _G_l_blur(),
      _img_buffer()
  { }

  void
  GaussianPyramid::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _G[0].create(n_rows, n_cols, CV_32F);
    _G_l_blur.create(n_rows, n_cols, CV_32F);
    for (size_t i = 1; i < _G.size(); ++i)
    {
      _G[i].create(n_rows, n_cols, CV_32F);
    }
  }

  void
  GaussianPyramid::
  apply(cv::cuda::GpuMat const& image,
	float decimation_ratio,
	float sigma)
  {
    if (decimation_ratio <= 1)
      throw std::runtime_error("Decimation ratio should be larger than 1");

    float n_rows    = image.rows;
    float n_cols    = image.cols;
    auto filter = cv::cuda::createGaussianFilter(
      CV_32F, CV_32F, cv::Size(5, 5), static_cast<double>(sigma));

    image.copyTo(_G[0]);
    for (size_t l = 0; l < _G.size()-1; ++l)
    {
      float lth_dec_ratio = powf(decimation_ratio, l+1);
      size_t M_dec        = static_cast<size_t>(ceil(n_rows/lth_dec_ratio));
      size_t N_dec        = static_cast<size_t>(ceil(n_cols/lth_dec_ratio));

      filter->apply(_G[l], _G_l_blur);
      cv::cuda::resize(_G_l_blur, _G[l+1], cv::Size(N_dec, M_dec), cv::INTER_NEAREST);
    }
  }

  void
  GaussianPyramid::
  apply(cv::Mat const& image,
	float decimation_ratio,
	float sigma)
  {
    _img_buffer.upload(image);
    this->apply(_img_buffer, decimation_ratio, sigma);
  }
}
