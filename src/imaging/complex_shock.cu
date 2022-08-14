
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

#include "complex_shock.hpp"

#include "utils.hpp"
#include "cuda_utils.hpp"

#include <cuComplex.h>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>

#include <cmath>

namespace usdg
{
  __global__ void
  cplxshock(int M, int N,
	    cv::cuda::PtrStepSz<float2>  const img_src,
	    cv::cuda::PtrStepSz<uchar>   const mask,
	    float2 lambda,
	    float lambda_tilde,
	    float theta,
	    float a,
	    float dt, 
	    cv::cuda::PtrStepSz<float2> img_dst)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    int xp = i + 1;
    int xm = i - 1;
    int yp = j + 1;
    int ym = j - 1;

    float2 u_c    = img_src(i,j);
    float2 u_xp   = fetch_pixel(img_src, xp,  j, mask, u_c);
    float2 u_xm   = fetch_pixel(img_src, xm,  j, mask, u_c);
    float2 u_yp   = fetch_pixel(img_src,  i, yp, mask, u_c);
    float2 u_ym   = fetch_pixel(img_src,  i, ym, mask, u_c);
    float2 u_xpyp = fetch_pixel(img_src, xp, yp, mask, u_c);
    float2 u_xmyp = fetch_pixel(img_src, xm, yp, mask, u_c);
    float2 u_xpym = fetch_pixel(img_src, xp, ym, mask, u_c);
    float2 u_xmym = fetch_pixel(img_src, xm, ym, mask, u_c);

    float2 Dx_u   = 0.5f*(u_xp - u_xm);
    float2 Dy_u   = 0.5f*(u_yp - u_ym);
    float2 Dxx_u  = u_xp + u_xm - 2*u_c;
    float2 Dyy_u  = u_yp + u_ym - 2*u_c;
    float2 Dxy_u  = 0.5f*(0.5f*(u_xpyp - u_xmyp) - 0.5f*(u_xpym - u_xmym));
      
    float2 t1 = Dxx_u*Dx_u*Dx_u + Dyy_u*Dy_u*Dy_u;
    float2 t2 = Dxy_u*Dx_u*Dy_u;
    float2 t3 = Dx_u*Dx_u + Dy_u*Dy_u;
    t3.x     += 1e-5;
    t3.y     += 1e-5;

    float2 Detaeta_u = cuCdivf(t1 + 2*t2, t3);
    float2 Dxixi_u   = cuCdivf(t1 - 2*t2, t3);
  
    float Dx_u_fluxlim = minmod(u_xp - u_c, u_c - u_xm);
    float Dy_u_fluxlim = minmod(u_yp - u_c, u_c - u_ym);
    float D_u_norm = sqrt(Dx_u_fluxlim*Dx_u_fluxlim + Dy_u_fluxlim*Dy_u_fluxlim);

    float const pi = 3.141592654f;

    float2 lambda_Detaeta_u     = lambda*Detaeta_u;
    float2 lambda_tilde_Dxixi_u = lambda_tilde*Dxixi_u;

    img_dst(i,j).x = img_src(i,j).x + dt*(
      -2/pi*atan(a/theta*u_c.y)*D_u_norm
      + lambda_Detaeta_u.x
      + lambda_tilde_Dxixi_u.x);

    img_dst(i,j).y = img_src(i,j).y + dt*(
      lambda_Detaeta_u.y
      + lambda_tilde_Dxixi_u.y);
  }

  ComplexShock::
  ComplexShock()
    : _img_buf1(),
    _img_buf2(),
    _img_in_buf(),
    _img_out_buf(),
    _mask_buf()
  {}

  void
  ComplexShock::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _img_buf1.create(   n_rows, n_cols, CV_32FC2);
    _img_buf2.create(   n_rows, n_cols, CV_32FC2);
    _img_in_buf.create( n_rows, n_cols, CV_32F);
    _img_out_buf.create(n_rows, n_cols, CV_32F);
    _mask_buf.create(   n_rows, n_cols, CV_8U);
  }

  void
  ComplexShock::
  apply(cv::cuda::GpuMat const& image,
	cv::cuda::GpuMat const& mask,
	cv::cuda::GpuMat&       output,
	float r,
	float lambda_tilde,
	float a,
	float theta,
	float dt, int n_iters)
  {
    cv::cuda::GpuMat planes[] = {
      cv::cuda::GpuMat(image),
      cv::cuda::GpuMat(image.size(), CV_32F, cv::Scalar(0.f))};
    cv::cuda::merge(planes, 2, _img_buf1);

    float2 lambda;
    lambda.x = r*cosf(theta);
    lambda.y = r*sinf(theta);

    size_t M = static_cast<size_t>(image.rows);
    size_t N = static_cast<size_t>(image.cols);
    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    for (size_t i = 0; i < n_iters; ++i)
    {
      usdg::cplxshock<<<grid, block>>>(
	M, N, _img_buf1, mask,
	lambda, lambda_tilde, theta, a, dt, _img_buf2);
      cv::swap(_img_buf1, _img_buf2);
    }
    cuda_check( cudaPeekAtLastError() );
    cv::cuda::split(_img_buf1, planes);
    planes[0].copyTo(output);
  }

  void
  ComplexShock::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	cv::Mat&       output,
	float r,
	float lambda_tilde,
	float a,
	float theta,
	float dt, int n_iters)
  {
    _img_in_buf.upload(image);
    _mask_buf.upload(  mask);
    this->apply(_img_in_buf, _mask_buf, _img_out_buf,
		r, lambda_tilde, a, theta,
		dt, n_iters);
    _img_out_buf.download(output);
  }
}