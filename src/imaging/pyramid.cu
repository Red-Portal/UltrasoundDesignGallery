
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

#include "pyramid.hpp"
#include "cuda_utils.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <stdexcept>
#include <iostream>

#include <cmath>

namespace usdg
{
  void
  LaplacianPyramid::
  compute_laplacian_pyramid(cv::Mat const& image,
			    cv::Mat const& mask,
			    cv::Mat& blur_buffer,
			    std::vector<cv::Mat>& G_buffer,
			    std::vector<cv::Mat>& L_out,
			    size_t n_levels,
			    float decimation_ratio,
			    float sigma) const
  {
    image.copyTo(G_buffer[0]);
    size_t M = image.rows;
    size_t N = image.cols;
    for (size_t i = 0; i < n_levels; ++i)
    {
      size_t M_dec = static_cast<size_t>(ceil(M/pow(decimation_ratio, i+1)));
      size_t N_dec = static_cast<size_t>(ceil(N/pow(decimation_ratio, i+1)));
      cv::GaussianBlur(G_buffer[i], blur_buffer, cv::Size(7, 7), sigma, sigma);
      cv::resize(blur_buffer, G_buffer[i+1], cv::Size(N_dec, M_dec));
      L_out[i] = G_buffer[i] - blur_buffer;
    }
  }

  __global__ void
  local_laplacian(int M, int N,
		  cv::cuda::PtrStepSzf       const I,
		  cv::cuda::PtrStepSzf       const G,
		  cv::cuda::PtrStepSz<uchar> const mask,
		  cv::cuda::PtrStepSzf             remap,
		  float    dec_rate,
		  float    alpha,
		  float    beta,
		  float    sigma)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N)// || mask(i,j) == 0)
      return;

    int i_remap = floor(static_cast<float>(i)/dec_rate);
    int j_remap = floor(static_cast<float>(j)/dec_rate);
    float g     = G(i_remap, j_remap);
    float p     = I(i, j);
    float delta = p - g;
    if (fabs(delta) <= sigma)
      remap(i,j) = g + usdg::sign(delta)*sigma*pow(fabs(delta) / sigma, alpha);
    else
      remap(i,j) = g + usdg::sign(delta)*(beta*(fabs(delta) - sigma) + sigma);
  }

  LaplacianPyramid::
  LaplacianPyramid(size_t n_scales)
    : _L(n_scales),
      _G(n_scales),
      _L_buffer(n_scales),
      _G_buffer(n_scales),
      _masks(n_scales),
      _blur_buffer(),
      _remap_buffer()
  { }

  void
  LaplacianPyramid::
  preallocate(size_t n_rows, size_t n_cols)
  {
    for (size_t i = 0; i < _G.size(); ++i)
    {
      _G[i].create(       n_rows, n_cols, CV_32F);
      _L[i].create(       n_rows, n_cols, CV_32F);
      _G_buffer[i].create(n_rows, n_cols, CV_32F);
      _L_buffer[i].create(n_rows, n_cols, CV_32F);
      _masks[i].create(   n_rows, n_cols, CV_32F);
    }
    _blur_buffer.create( n_rows, n_cols, CV_32F);
    _remap_buffer.create(n_rows, n_cols, CV_32F);

    _img_device_buffer.create(  n_rows, n_cols, CV_32F);
    _mask_device_buffer.create( n_rows, n_cols, CV_32F);
    _G_device_buffer.create(    n_rows, n_cols, CV_32F);
    _remap_device_buffer.create(n_rows, n_cols, CV_32F);
  }

  void
  LaplacianPyramid::
  apply(cv::Mat const& img,
	cv::Mat const& mask,
	float decimation_ratio,
	float dec_sigma,
	float alpha,
	float beta,
	float sigma)
  {
    if (decimation_ratio <= 1)
      throw std::runtime_error("Decimation ratio should be larger than 1");

    size_t M = img.rows;
    size_t N = img.cols;

    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    _img_device_buffer.upload(img);
    _mask_device_buffer.upload(mask);

    img.copyTo(_G[0]);
    mask.copyTo(_masks[0]);

    for (size_t i = 0; i < _G.size()-1; ++i)
    {
      float total_dec_ratio = pow(decimation_ratio, i+1);
      size_t M_dec = static_cast<size_t>(ceil(M/total_dec_ratio));
      size_t N_dec = static_cast<size_t>(ceil(N/total_dec_ratio));
      cv::GaussianBlur(_G[i],  _blur_buffer, cv::Size(7, 7), dec_sigma, dec_sigma);
      cv::resize(_blur_buffer, _G[i+1],      cv::Size(N_dec, M_dec));
      cv::resize(mask,         _masks[i+1],  cv::Size(N_dec, M_dec));

      auto roi   = cv::Rect(0, 0, N_dec, M_dec);
      auto G_roi = _G_device_buffer(roi);
      G_roi.upload(_G[i+1]);
      usdg::local_laplacian<<<grid, block>>>(
	M, N,
	_img_device_buffer,
	_G_device_buffer,
	_mask_device_buffer,
	_remap_device_buffer,
	total_dec_ratio,
	alpha, beta, sigma);
      _remap_device_buffer.download(_remap_buffer);

      this->compute_laplacian_pyramid(_remap_buffer, mask, _blur_buffer,
				      _G_buffer, _L_buffer,
				      i+1,
				      decimation_ratio,
				      dec_sigma);
      _L_buffer[i].copyTo(_L[i]);
    }
    _G.back().copyTo(_L.back());
  }
}
