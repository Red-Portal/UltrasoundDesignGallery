
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

#include <cmath>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include "osrad.hpp"
#include "utils.hpp"
#include "cuda_utils.hpp"

namespace usdg
{
  __global__ void
  osrad_diffusion_matrix(cv::cuda::PtrStepSzf const img_smoothed,
			 cv::cuda::PtrStepSzf Dxx,
			 cv::cuda::PtrStepSzf Dxy,
			 cv::cuda::PtrStepSzf Dyy,
			 float sigma_g,
			 float ctang)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = img_smoothed.rows;
    int N = img_smoothed.cols;

    if (i >= M || j >= N)
      return;

    int xp = min(i+1, M-1);
    int xm = max(i-1,   0);
    int yp = min(j+1, N-1);
    int ym = max(j-1,   0);

    float g_x   = (img_smoothed(xp,  j) - img_smoothed(xm,  j))/2;
    float g_y   = (img_smoothed( i, yp) - img_smoothed( i, ym))/2;
    float g_mag = sqrt(g_x*g_x + g_y*g_y);

    float e0_x = 0.0;
    float e0_y = 0.0;
    if (g_mag > 1e-7f)
    {
      e0_x = g_x / g_mag;
      e0_y = g_y / g_mag;
    }
    else
    {
      e0_x = 1.0;
      e0_y = 0.0;
    }

    float e1_x = -e0_y;
    float e1_y = e0_x;

    float lambda0 = usdg::tukey_biweight(g_mag, sigma_g);
    float lambda1 = ctang;

    Dxx(i,j) = lambda0*e0_x*e0_x + lambda1*e1_x*e1_x;
    Dxy(i,j) = lambda0*e0_x*e0_y + lambda1*e1_x*e1_y;
    Dyy(i,j) = lambda0*e0_y*e0_y + lambda1*e1_y*e1_y;
  }

  __global__ void
  osrad_diffusion(cv::cuda::PtrStepSzf const img_src,
		  cv::cuda::PtrStepSzf const img_smoothed,
		  cv::cuda::PtrStepSzf const Dxx,
		  cv::cuda::PtrStepSzf const Dxy,
		  cv::cuda::PtrStepSzf const Dyy,
		  cv::cuda::PtrStepSzf img_dst,
		  float dt,
		  float sigma_r)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = img_src.rows;
    int N = img_src.cols;

    if (i >= M || j >= N)
      return;

    int xp = min(i+1, M-1);
    int xm = max(i-1,   0);
    int yp = min(j+1, N-1);
    int ym = max(j-1,   0);
    
    float u1 = img_src(xm, yp);
    float u2 = img_src( i, yp);
    float u3 = img_src(xp, yp);
    float u4 = img_src(xm,  j);
    float u5 = img_src( i,  j);
    float u6 = img_src(xp,  j);
    float u7 = img_src(xm, ym);
    float u8 = img_src( i, ym);
    float u9 = img_src(xp, ym);

    float r1 = exp(-img_smoothed(xm, yp)/sigma_r);
    float r2 = exp(-img_smoothed(i,  yp)/sigma_r);
    float r3 = exp(-img_smoothed(xp, yp)/sigma_r);
    float r4 = exp(-img_smoothed(xm, j )/sigma_r);
    float r6 = exp(-img_smoothed(xp, j )/sigma_r);
    float r7 = exp(-img_smoothed(xm, ym)/sigma_r);
    float r8 = exp(-img_smoothed(i,  ym)/sigma_r);
    float r9 = exp(-img_smoothed(xp, ym)/sigma_r);

    float eps = 1e-7;
    float c1 = max((1.0/4)*(Dxy(xm,  j) - Dxy(i, yp))*r1, eps);
    float c2 = max((1.0/2)*(Dyy( i, yp) + Dyy(i,  j))*r2, eps);
    float c3 = max((1.0/4)*(Dxy(xp,  j) + Dxy(i, yp))*r3, eps);
    float c4 = max((1.0/2)*(Dxx(xm,  j) + Dxx(i,  j))*r4, eps);
    float c6 = max((1.0/2)*(Dxx(xp,  j) + Dxx(i,  j))*r6, eps);
    float c7 = max((1.0/4)*(Dxy(xm,  j) + Dxy(i, ym))*r7, eps);
    float c8 = max((1.0/2)*(Dyy( i, ym) + Dyy(i,  j))*r8, eps);
    float c9 = max((1.0/4)*(Dxy(xp,  j) - Dxy(i, ym))*r9, eps);

    img_dst(i,j) = (u5 + dt*(c1*u1 + c2*u2 + c3*u3 + c4*u4 + c6*u6 + c7*u7 + c8*u8 + c9*u9)) /
      (1 + dt*(c1 + c2 + c3 + c4 + c6 + c7 + c8 + c9));
  }

  void
  osrad(cv::cuda::GpuMat& G_buf1,
	cv::cuda::GpuMat& G_buf2,
	cv::cuda::GpuMat& L_buf1,
	cv::cuda::GpuMat& L_buf2,
	cv::cuda::GpuMat& Dxx_buf,
	cv::cuda::GpuMat& Dxy_buf,
	cv::cuda::GpuMat& Dyy_buf,
	cv::Ptr<cv::cuda::Filter>& gaussian_filter,
	float dt,
	float sigma_r,
	float sigma_g,
	float ctang,
	size_t niters)
  /*
   * Perona, Pietro, and Jitendra Malik. 
   * "Scale-space and edge detection using anisotropic diffusion." 
   * IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 1990.
   */
  {
    size_t M  = static_cast<size_t>(G_buf1.rows);
    size_t N  = static_cast<size_t>(G_buf1.cols);
    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));
    gaussian_filter->apply(G_buf1, G_buf2);
    for (size_t i = 0; i < niters; ++i)
    {
      usdg::osrad_diffusion_matrix<<<grid,block>>>(L_buf2,
						   Dxx_buf,
						   Dxy_buf,
						   Dyy_buf,
						   sigma_g,
						   ctang);
      usdg::osrad_diffusion<<<grid,block>>>(L_buf1,
					    G_buf2,
					    Dxx_buf,
					    Dxy_buf,
					    Dyy_buf,
					    L_buf2,
					    dt,
					    sigma_r);
      cv::swap(L_buf1, L_buf2);
    }
    cuda_check( cudaPeekAtLastError() );
  }

  OSRAD::
  OSRAD()
    : _G_buf1(),
    _G_buf2(),
    _L_buf1(),
    _L_buf2(),
    _Dxx_buf(),
    _Dxy_buf(),
    _Dyy_buf(),
    _gaussian_filter()
  {}

  void
  OSRAD::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _G_buf1.create(n_rows, n_cols, CV_32F);
    _G_buf2.create(n_rows, n_cols, CV_32F);
    _L_buf1.create(n_rows, n_cols, CV_32F);
    _L_buf2.create(n_rows, n_cols, CV_32F);
    _Dxx_buf.create(n_rows, n_cols, CV_32F);
    _Dxy_buf.create(n_rows, n_cols, CV_32F);
    _Dyy_buf.create(n_rows, n_cols, CV_32F);
    _gaussian_filter = cv::cuda::createGaussianFilter(
      CV_32F, CV_32F, cv::Size(5,5), 4.0);
  }

  void
  OSRAD::
  apply(cv::Mat const& G_image,
	cv::Mat const& L_image,
	cv::Mat&       output,
	float dt,
	float sigma_r,
	float sigma_g,
	float ctang,
	size_t niters)
  {
    _G_buf1.upload(G_image);
    _L_buf1.upload(L_image);
    usdg::osrad(_G_buf1, _G_buf2,
		_L_buf1, _L_buf2,
		_Dxx_buf,
		_Dxy_buf,
		_Dyy_buf,
		_gaussian_filter,
		dt, sigma_r, sigma_g, ctang, niters);
    _L_buf2.download(output);
  }
}