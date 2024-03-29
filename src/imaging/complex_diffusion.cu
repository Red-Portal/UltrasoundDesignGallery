

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

#include "complex_diffusion.hpp"

#include "utils.hpp"
#include "cuda_utils.hpp"

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>

#include <cmath>

namespace usdg
{
  __global__ void
  rpncd_compute_diffusivity(int M, int N,
			    cv::cuda::PtrStepSz<float2> const img,
			    cv::cuda::PtrStepSz<uchar>  const mask,
			    cv::cuda::PtrStepSz<float2>       coeff,
			    float k, float theta)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    float2 u = img(i,j);

    float ratio    = u.y / (k*theta);
    float amp      = 1 / (1 + ratio*ratio);
    float coeff_re = amp*cosf(theta);
    float coeff_im = amp*sinf(theta);

    coeff(i,j).x = coeff_re;
    coeff(i,j).y = coeff_im;
  }

  __global__ void
  rpncd_diffuse(int M, int N,
		cv::cuda::PtrStepSz<float2>  const img_src,
		cv::cuda::PtrStepSz<float2>  const coeff,
		cv::cuda::PtrStepSz<uchar>   const mask,
		float dt, 
		cv::cuda::PtrStepSz<float2>        img_dst)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    int xp = min(i+1, M-1);
    int xm = max(i-1,   0);
    int yp = min(j+1, N-1);
    int ym = max(j-1,   0);

    float2 I_c  = img_src(i,j);
    float2 I_xp = fetch_pixel(img_src, xp,  j, mask, I_c);
    float2 I_xm = fetch_pixel(img_src, xm,  j, mask, I_c);
    float2 I_yp = fetch_pixel(img_src,  i, yp, mask, I_c);
    float2 I_ym = fetch_pixel(img_src,  i, ym, mask, I_c);

    float2 C_c  = coeff(i,j);
    float2 C_xp = fetch_pixel(coeff, xp,  j, mask, I_c);
    float2 C_xm = fetch_pixel(coeff, xm,  j, mask, I_c);
    float2 C_yp = fetch_pixel(coeff,  i, yp, mask, I_c);
    float2 C_ym = fetch_pixel(coeff,  i, ym, mask, I_c);

    img_dst(i,j) = I_c + (
      dt/4*((I_xp - I_c)*C_xp
	    + (I_xm - I_c)*C_xm
	    + (I_yp - I_c)*C_yp
	    + (I_ym - I_c)*C_ym));
  }

  ComplexDiffusion::
  ComplexDiffusion()
    : _img_buf1(),
    _img_buf2(),
    _coeff(),
    _img_in_buf(),
    _img_out_buf(),
    _mask_buf()
  {}

  void
  ComplexDiffusion::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _img_buf1.create(   n_rows, n_cols, CV_32FC2);
    _img_buf2.create(   n_rows, n_cols, CV_32FC2);
    _coeff.create(      n_rows, n_cols, CV_32FC2);
    _img_in_buf.create( n_rows, n_cols, CV_32F);
    _img_out_buf.create(n_rows, n_cols, CV_32F);
    _mask_buf.create(   n_rows, n_cols, CV_8U);
  }

  void
  ComplexDiffusion::
  apply(cv::cuda::GpuMat const& image,
	cv::cuda::GpuMat const& mask,
	cv::cuda::GpuMat&       output,
	float k, float theta, 
	float dt, int n_iters)
  {
    cv::cuda::GpuMat planes[] = {
      cv::cuda::GpuMat(image),
      cv::cuda::GpuMat(image.size(), CV_32F, cv::Scalar(0.f))};
    cv::cuda::merge(planes, 2, _img_buf1);

    size_t M = static_cast<size_t>(image.rows);
    size_t N = static_cast<size_t>(image.cols);
    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    for (size_t i = 0; i < n_iters; ++i)
    {
      usdg::rpncd_compute_diffusivity<<<grid, block>>>(M, N, _img_buf1, mask, _coeff, k, theta);
      usdg::rpncd_diffuse<<<grid, block>>>(M, N, _img_buf1, _coeff, mask, dt, _img_buf2);
      cv::swap(_img_buf1, _img_buf2);
    }
    cuda_check( cudaPeekAtLastError() );
    cv::cuda::split(_img_buf1, planes);
    planes[0].copyTo(output, mask);
  }

  void
  ComplexDiffusion::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	cv::Mat&       output,
	float k, float theta, 
	float dt, int n_iters)
  {
    _img_in_buf.upload(image);
    _mask_buf.upload(  mask);
    this->apply(_img_in_buf, _mask_buf, _img_out_buf,
		k, theta, dt, n_iters);
    _img_out_buf.download(output);
  }
}