
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
#include <iostream>

namespace usdg
{
  Pipeline::
  Pipeline(size_t n_rows, size_t n_cols)
    : _pyramid(4),
      _laplace(),
      _ncd(),
      _rpncd()
  {
    _laplace.preallocate(n_rows, n_cols);
    _ncd.preallocate(    n_rows, n_cols);
    _rpncd.preallocate(  n_rows, n_cols);
  }

  void
  Pipeline::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	cv::Mat&       output,
	float laplace_beta,
	float laplace_sigmag,
	float ncd1_alpha,
	float ncd1_s,
	float ncd2_alpha,
	float ncd2_s,
	float rpncd_k)
  {
    float const rate = 2.0;
    float const laplace_alpha = 0.5;
    _pyramid.apply(image, mask, rate, rate/2);

    _laplace.apply(_pyramid.G(0),
		   _pyramid.mask(0),
		   _pyramid.L(0),
		   laplace_alpha,
		   laplace_beta,
		   laplace_sigmag);

    _laplace.apply(_pyramid.G(1),
		   _pyramid.mask(1),
		   _pyramid.L(1),
		   laplace_alpha,
		   laplace_beta,
		   laplace_sigmag);

    _laplace.apply(_pyramid.G(2),
		   _pyramid.mask(2),
		   _pyramid.L(2),
		   laplace_alpha,
		   laplace_beta,
		   laplace_sigmag);

    /* Pyarmid level 3 is left unchanged */

    /* Pyarmid level 2 denoising and synthesis */
    auto& L2 = _pyramid.L(2);
    auto G2  = cv::Mat();
    cv::resize(_pyramid.L(3), G2, cv::Size(L2.cols, L2.rows));
    G2 += L2;
    _ncd.apply(G2, _pyramid.mask(2), G2, 2.0f, ncd2_alpha, ncd2_s, 2.0f, 30);

    /* Pyarmid level 1 denoising and synthesis */
    auto& L1 = _pyramid.L(1);
    auto G1  = cv::Mat();
    cv::resize(G2, G1, cv::Size(L1.cols, L1.rows));
    G1 += L1;
    _ncd.apply(G1, _pyramid.mask(1), G1, 2.0f, ncd1_alpha, ncd1_s, 2.0f, 30);

    /* Pyarmid level 0 denoising and synthesis */
    auto& L0 = _pyramid.L(0);
    auto G0  = cv::Mat();
    cv::resize(G1, G0, cv::Size(L0.cols, L0.rows));
    G0 += L0;
    float const theta = 5.f/180.f*3.141592;

    //_ncd.apply(G0, _pyramid.mask(0), G0, 1.0f, ncd1_alpha, ncd1_s, 2.0f, 30);
    _rpncd.apply(G0, _pyramid.mask(0), G0, rpncd_k, theta, 0.1f, 30.f);

    G0.copyTo(output);
  }
}
