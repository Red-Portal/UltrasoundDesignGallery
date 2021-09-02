
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
#include <opencv4/opencv2/highgui.hpp>

#include <algorithm>

namespace usdg
{
  Pipeline::
  Pipeline(size_t n_rows, size_t n_cols)
    : _pyramid(4),
      _ncd(),
      _rpncd(),
     _edge_enhance()
  {
    //_pyramid.preallocate(n_rows, n_cols);
    _ncd.preallocate(   n_rows, n_cols);
    _rpncd.preallocate( n_rows, n_cols);
    _edge_enhance.preallocate(n_rows, n_cols);
  }

  void
  Pipeline::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	cv::Mat&       output,
	float ee1_beta,
	float ee1_sigma,
	float ee2_beta,
	float ee2_sigma,
	float ncd1_s,
	float ncd1_alpha,
	float ncd2_s,
	float ncd2_alpha,
	float rpncd_k)
  {
    float const rate = 2.0;
    _pyramid.apply(image, mask, rate, rate/2);


    /* Pyramid level 3 is left unchanged */

    /* Pyramid level 2 denoising and synthesis */
    auto& L2 = _pyramid.L(2);
    auto G2  = cv::Mat();
    cv::resize(_pyramid.L(3), G2, cv::Size(L2.cols, L2.rows));
    G2 += L2;
    G2 *= 255;
    _ncd.apply(G2, _pyramid.mask(2), G2, 2.f, ncd1_alpha, ncd1_s, 2.0f, 10);
    G2 /= 255;

    float const theta = 5.f/180.f*3.141592;

    /* Pyramid level 1 denoising and synthesis */
    auto& L1 = _pyramid.L(1);
    auto G1  = cv::Mat();
    cv::resize(G2, G1, cv::Size(L1.cols, L1.rows));
    _edge_enhance.apply(L1, _pyramid.mask(1), 2.0, ee1_beta, ee1_sigma);
    G1 += L1;
    G1 *= 255;
    _ncd.apply(G1, _pyramid.mask(1), G1, 2.f, ncd2_alpha, ncd2_s, 2.0f, 10);
    G1 /= 255;
    //_rpncd.apply(G1, _pyramid.mask(1), G1, rpncd_k1, theta, 0.3f, 20.f);

    /* Pyramid level 0 denoising and synthesis */
    auto& L0 = _pyramid.L(0);
    auto G0  = cv::Mat();
    cv::resize(G1, G0, cv::Size(L0.cols, L0.rows));
    _edge_enhance.apply(L0, _pyramid.mask(0), 2.0, ee2_beta, ee2_sigma);
    G0 += L0;
    _rpncd.apply(G0, _pyramid.mask(0), G0, rpncd_k, theta, 0.3f, 20.f);

    G0.copyTo(output);
  }
}
