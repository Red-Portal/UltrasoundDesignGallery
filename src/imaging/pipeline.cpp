
/*
 * Copyright (C) 2021  Ray Kim
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

#include "pipeline.hpp"

#include <algorithm>
#include <iostream>

namespace usdg
{
  Pipeline::
  Pipeline(size_t n_rows, size_t n_cols)
    : _diffusion(),
      _despeckle_buf(cv::Mat(static_cast<int>(n_rows), static_cast<int>(n_cols), CV_32F)),
      _logimage_buf( cv::Mat(static_cast<int>(n_rows), static_cast<int>(n_cols), CV_32F)),
      _expimage_buf( cv::Mat(static_cast<int>(n_rows), static_cast<int>(n_cols), CV_32F)),
      _lopass_buf(   cv::Mat(static_cast<int>(n_rows), static_cast<int>(n_cols), CV_32F)),
      _hipass_buf(   cv::Mat(static_cast<int>(n_rows), static_cast<int>(n_cols), CV_32F))
  {
    _diffusion.preallocate(n_rows, n_cols);
  }

  void
  Pipeline::
  apply(cv::Mat const& image,
	cv::Mat&       output,
	float t,     float ts_a, float ts_b, float sigma_g,
	float ctang, float theta, float alpha, float beta)
  {
    float dt     = 0.3;
    size_t niter = static_cast<size_t>(ceil(t / dt));
    _diffusion.apply(image, _despeckle_buf, dt, ts_a, ts_b, sigma_g, ctang, niter);

    _despeckle_buf += 0.01f;
    cv::log(_despeckle_buf, _logimage_buf);
    cv::GaussianBlur(_logimage_buf, _lopass_buf, cv::Size(5, 5), theta);
    cv::subtract(_logimage_buf, _lopass_buf, _hipass_buf);
    _lopass_buf *= alpha;
    _hipass_buf *= beta;
    _lopass_buf += _hipass_buf;
    cv::exp(_lopass_buf, _expimage_buf);
    _expimage_buf -= 0.01f;

    for (int i = 0; i < _expimage_buf.rows; ++i) {
      for (int j = 0; j < _expimage_buf.cols; ++j) {
	 _expimage_buf.at<float>(i,j) = std::clamp(
	  _expimage_buf.at<float>(i,j), 0.0f, 1.0f);
      }
    }
    output = _expimage_buf;
  }
}
