
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

#include "lowpass.hpp"

#include <opencv4/opencv2/highgui.hpp>

#include <vector>
#include <iostream>

#include <cmath>
#include <cassert>

namespace usdg
{
  inline void
  circshift(cv::Mat &out, const cv::Point &delta)
  /* 
   * Original by @TerryBryant
   * First created in Sep. 25th, 2017
   * add fft2 and ifft2 in Oct. 06th, 2017
   */
  {
    cv::Size sz = out.size();

    // error checking
    assert(sz.height > 0 && sz.width > 0);

    // no need to shift
    if ((sz.height == 1 && sz.width == 1) || (delta.x == 0 && delta.y == 0))
      return;

    // delta transform
    int x = delta.x;
    int y = delta.y;
    if (x > 0) x = x % sz.width;
    if (y > 0) y = y % sz.height;
    if (x < 0) x = x % sz.width + sz.width;
    if (y < 0) y = y % sz.height + sz.height;


    // in case of multiple dimensions
    std::vector<cv::Mat> planes;
    split(out, planes);

    for (size_t i = 0; i < planes.size(); i++)
    {
      // vertical
      cv::Mat tmp0, tmp1, tmp2, tmp3;
      cv::Mat q0(planes[i], cv::Rect(0, 0, sz.width, sz.height - y));
      cv::Mat q1(planes[i], cv::Rect(0, sz.height - y, sz.width, y));
      q0.copyTo(tmp0);
      q1.copyTo(tmp1);
      tmp0.copyTo(planes[i](cv::Rect(0, y, sz.width, sz.height - y)));
      tmp1.copyTo(planes[i](cv::Rect(0, 0, sz.width, y)));

      // horizontal
      cv::Mat q2(planes[i], cv::Rect(0, 0, sz.width - x, sz.height));
      cv::Mat q3(planes[i], cv::Rect(sz.width - x, 0, x, sz.height));
      q2.copyTo(tmp2);
      q3.copyTo(tmp3);
      tmp2.copyTo(planes[i](cv::Rect(x, 0, sz.width - x, sz.height)));
      tmp3.copyTo(planes[i](cv::Rect(0, 0, x, sz.height)));
    }

    cv::merge(planes, out);
  }

  inline void
  fftshift(cv::Mat &out)
  /* 
   * Original by @TerryBryant
   * First created in Sep. 25th, 2017
   * add fft2 and ifft2 in Oct. 06th, 2017
   */
  {
    cv::Size sz = out.size();
    cv::Point pt(0, 0);
    pt.x = (int)floor(sz.width / 2.0);
    pt.y = (int)floor(sz.height / 2.0);
    usdg::circshift(out, pt);
  }

  inline void
  ifftshift(cv::Mat &out)
  /* 
   * Original by @TerryBryant
   * First created in Sep. 25th, 2017
   * add fft2 and ifft2 in Oct. 06th, 2017
   */
  {
    cv::Size sz = out.size();
    cv::Point pt(0, 0);
    pt.x = (int)ceil(sz.width / 2.0);
    pt.y = (int)ceil(sz.height / 2.0);
    usdg::circshift(out, pt);
  }

  inline void
  butterworth(cv::Mat& kernel,
	      float cutoff_theta,
	      float order)
  {
    size_t M      = kernel.rows;
    size_t N      = kernel.cols;
    size_t M_half = M/2;
    size_t N_half = N/2;

    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
	float theta_x  = (static_cast<double>(i) + 0.5 - M_half) / (M-1);
	float theta_y  = (static_cast<double>(j) + 0.5 - N_half) / (N-1);
	float r        = sqrt(theta_x*theta_x + theta_y*theta_y);
	float response = 1 / (1 + pow(r / (cutoff_theta / 2), 2*order));
	kernel.at<cv::Complex<float>>(i,j) = cv::Complex<float>(response, 0);
      }
    }
    usdg::ifftshift(kernel);
  }

  ButterworthLPF::
  ButterworthLPF()
    : _fft_size(),
      _image_padded(     _fft_size, CV_32F),
      _image_spectrum(   _fft_size, CV_32FC2),
      _kernel_spectrum(  _fft_size, CV_32FC2),
      _filtered_spectrum(_fft_size, CV_32FC2),
      _filtered(         _fft_size, CV_32FC2)
  { }

  void
  ButterworthLPF::
  preallocate(size_t M, size_t N)
  {
    _fft_size	       = cv::Size(cv::getOptimalDFTSize(N),
				  cv::getOptimalDFTSize(M));
    _image_padded      = cv::Mat(_fft_size, CV_32F);
    _image_spectrum    = cv::Mat(_fft_size, CV_32FC2);
    _kernel_spectrum   = cv::Mat(_fft_size, CV_32FC2);
    _filtered_spectrum = cv::Mat(_fft_size, CV_32FC2);
    _filtered	       = cv::Mat(_fft_size, CV_32FC2);
  }
  
  void
  ButterworthLPF::
  apply(cv::Mat const& img,
	cv::Mat& out,
	float cutoff_theta,
	float order)
  {
    float cutoff_adj = static_cast<float>(std::min(img.rows, img.cols)) /
      std::min(_fft_size.width, _fft_size.height) * cutoff_theta;
    usdg::butterworth(_kernel_spectrum, cutoff_theta, order);

    auto roi = cv::Mat(_image_padded, cv::Rect(0, 0, img.cols, img.rows));
    img.copyTo(roi);

    cv::dft(_image_padded,
	    _image_spectrum,
	    cv::DFT_COMPLEX_OUTPUT,
	    _image_padded.rows);
    cv::mulSpectrums(_image_spectrum,
		     _kernel_spectrum,
		     _filtered_spectrum,
		     0);
    cv::dft(_filtered_spectrum,
	    _filtered,
	    cv::DFT_INVERSE + cv::DFT_SCALE);

    for (size_t i = 0; i < img.rows; ++i) {
      for (size_t j = 0; j < img.cols; ++j) {
	out.at<float>(i,j) = cv::abs(_filtered.at<cv::Complex<float>>(i,j));
      }
    }
  }
}
