
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

#include "logcompression.hpp"

#include <opencv4/opencv2/core/core.hpp>

#include <algorithm>

namespace usdg
{
  Pipeline::
  Pipeline(size_t n_rows, size_t n_cols)
    : _pyramid(4),
      _ncd(),
      _rpncd()
  {
    _pyramid.preallocate(n_rows, n_cols);
    _ncd.preallocate(    n_rows, n_cols);
    _rpncd.preallocate(  n_rows, n_cols);
  }

  void
  Pipeline::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	cv::Mat&       output,
	float ll_alpha,
	float ll_beta,
	float ll_sigma,
	float ncd_s,
	float ncd_alpha,
	float rpncd_k1,
	float rpncd_k2)
  {
    float const rate = 2.0;
    _pyramid.apply(image, mask, rate, rate/2, ll_alpha, ll_beta, ll_sigma);

    /* Pyramid level 3 is left unchanged */

    /* Pyramid level 2 denoising and synthesis */
    auto& L2 = _pyramid.L(2);
    auto G2  = cv::Mat();
    cv::resize(_pyramid.L(3), G2, cv::Size(L2.cols, L2.rows));
    G2 += L2;
    _ncd.apply(G2*255, _pyramid.mask(2), G2, 2.f, ncd_alpha, ncd_s, 2.0f, 30);
    G2 /= 255;

    float const theta = 5.f/180.f*3.141592;

    /* Pyramid level 1 denoising and synthesis */
    auto& L1 = _pyramid.L(1);
    auto G1  = cv::Mat();
    cv::resize(G2, G1, cv::Size(L1.cols, L1.rows));
    G1 += L1;
    //_ncd.apply(G1, _pyramid.mask(1), G1, 2.f, ncd1_alpha, ncd1_s, 2.0f, 30);
    _rpncd.apply(G1, _pyramid.mask(1), G1, rpncd_k1, theta, 0.3f, 10.f);

    /* Pyramid level 0 denoising and synthesis */
    auto& L0 = _pyramid.L(0);
    auto G0  = cv::Mat();
    cv::resize(G1, G0, cv::Size(L0.cols, L0.rows));
    G0 += L0;
    //_ncd.apply(G0, _pyramid.mask(0), G0, 1.0f, ncd1_alpha, ncd1_s, 2.0f, 30);
    _rpncd.apply(G0, _pyramid.mask(0), G0, rpncd_k2, theta, 0.3f, 10.f);

    G0.copyTo(output);
  }
}
