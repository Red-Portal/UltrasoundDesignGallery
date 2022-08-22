
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

#include "cascaded_pyramid.hpp"

#include "logcompression.hpp"

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudawarping.hpp>

#include <algorithm>
#include <iostream>
#include <chrono>

namespace usdg
{
  CascadedPyramid::
  CascadedPyramid(size_t n_rows, size_t n_cols)
    : _pyramid(4, 10),
      _cshock(),
      _ncd1(),
      _ncd2(),
      _rpncd()
  {
    _pyramid.preallocate(n_rows, n_cols);
    _cshock.preallocate( n_rows, n_cols);
    _ncd1.preallocate(   n_rows, n_cols);
    _ncd2.preallocate(   n_rows, n_cols);
    _rpncd.preallocate(  n_rows, n_cols);
    _img_in_buf.create(  n_rows, n_cols, CV_32F);
    _img_out_buf.create( n_rows, n_cols, CV_32F);
    _mask_buf.create(    n_rows, n_cols, CV_32F);
  }

  void
  CascadedPyramid::
  apply(cv::cuda::GpuMat const& image,
	cv::cuda::GpuMat const& mask,
	cv::cuda::GpuMat&       output,
	float llf_alpha, 
	float llf_beta, 
	float llf_sigma, 
	float cshock_a,
	float ncd1_alpha,
	float ncd1_s,
	float ncd2_alpha,
	float ncd2_s,
	float rpncd_k)
  {
    (void)cshock_a;
    _pyramid.apply(image, mask, llf_alpha, llf_beta, llf_sigma);

    /* Pyramid level 3  */
    auto S      = cv::cuda::GpuMat();
    auto& L3    = _pyramid.L(3);
    L3.copyTo(S);
    //_cshock.apply(S, _pyramid.mask(3), S, 0.1, 0.1, cshock_a, 0.01, 0.2, 20);

    /* Pyramid level 2 denoising and synthesis */
    auto& L2 = _pyramid.L(2);
    cv::cuda::resize(S, S, cv::Size(L2.cols, L2.rows));
    cv::cuda::add(S, L2, S);
    _ncd1.apply(S, _pyramid.mask(2), S, 3.0, ncd1_alpha, ncd1_s, 1.0, 10);

    /* Pyramid level 1 denoising and synthesis */
    auto& L1 = _pyramid.L(1);
    cv::cuda::resize(S, S, cv::Size(L1.cols, L1.rows));
    cv::cuda::add(S, L1, S);
    _ncd2.apply(S, _pyramid.mask(1), S, 3.0, ncd2_alpha, ncd2_s, 1.0, 10);

    /* Pyramid level 0 denoising and synthesis */
    auto& L0 = _pyramid.L(0);
    cv::cuda::resize(S, S, cv::Size(L0.cols, L0.rows));
    cv::cuda::add(S, L0, S);
    _rpncd.apply(S, _pyramid.mask(0), S, rpncd_k, 5./180*3.141592, 0.5, 10);

    S.copyTo(output, mask);
  }

  void
  CascadedPyramid::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	cv::Mat&       output,
	float llf_alpha, 
	float llf_beta, 
	float llf_sigma, 
	float cshock_a,
	float ncd1_alpha,
	float ncd1_s,
	float ncd2_alpha,
	float ncd2_s,
	float rpncd_k)
  {
    _img_in_buf.upload(image);
    _mask_buf.upload(  mask);
    this->apply(_img_in_buf, _mask_buf, _img_out_buf,
		llf_alpha,
		llf_beta,
		llf_sigma,
		cshock_a,
		ncd1_alpha,
		ncd1_s,
		ncd2_alpha,
		ncd2_s,
		rpncd_k);
    _img_out_buf.download(output);
  }
}
