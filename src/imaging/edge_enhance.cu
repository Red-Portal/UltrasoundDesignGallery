
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

#include "edge_enhance.hpp"

#include "cuda_utils.hpp"

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <cmath>

namespace usdg
{
  __global__ void
  edge_enhance_impl(int M, int N,
		    cv::cuda::PtrStepSzf             image,
		    cv::cuda::PtrStepSz<uchar> const mask,
		    float alpha, float beta, float sigma)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    
    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    float x = image(i,j);

    if (abs(x) <= sigma)
      image(i,j) = sign(x)*sigma*__powf(abs(x) / sigma, alpha);
    else
      image(i,j) = sign(x)*(beta*(abs(x) - sigma) - sigma);
  }

  EdgeEnhance::
  EdgeEnhance()
    : _image(), _mask()
  { }

  void 
  EdgeEnhance::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _image.create(n_rows, n_cols, CV_32F);
    _mask.create( n_rows, n_cols, CV_8U);
  }

  void
  EdgeEnhance::
  apply(cv::Mat& image,
	cv::Mat const& mask,
	float alpha, float beta, float sigma)
  {
    size_t M  = static_cast<size_t>(image.rows);
    size_t N  = static_cast<size_t>(image.cols);
    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    auto roi = cv::Rect(0, 0, N, M);

    auto roi_image = _image(roi);
    auto roi_mask  = _mask(roi);

    roi_image.upload(image);
    roi_mask.upload(mask);

    usdg::edge_enhance_impl<<<grid, block>>>(M, N, _image, _mask, alpha, beta, sigma);
    roi_image.download(image);
  }
}