
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

#include "tsonad.hpp"
#include "utils.hpp"
#include "cuda_utils.hpp"

namespace usdg
{
  __global__ void
  tsonad_diffusion_matrix(cv::cuda::PtrStepSzf const img_smoothed,
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


  __device__ __forceinline__ float
  kumaraswamy_cdf(float x, float a, float b)
  {
    return pow(1 - pow(x, a), b);
  }

  __global__ void
  tsonad_diffusion(cv::cuda::PtrStepSzf const img_src,
		   cv::cuda::PtrStepSzf const img_smoothed,
		   cv::cuda::PtrStepSzf const Dxx,
		   cv::cuda::PtrStepSzf const Dxy,
		   cv::cuda::PtrStepSzf const Dyy,
		   cv::cuda::PtrStepSzf img_dst,
		   float ts_a,
		   float ts_b,
		   float dt)
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

    float r1 = kumaraswamy_cdf(img_smoothed(xm, yp), ts_a, ts_b);
    float r2 = kumaraswamy_cdf(img_smoothed(i,  yp), ts_a, ts_b);
    float r3 = kumaraswamy_cdf(img_smoothed(xp, yp), ts_a, ts_b);
    float r4 = kumaraswamy_cdf(img_smoothed(xm, j ), ts_a, ts_b);
    float r6 = kumaraswamy_cdf(img_smoothed(xp, j ), ts_a, ts_b);
    float r7 = kumaraswamy_cdf(img_smoothed(xm, ym), ts_a, ts_b);
    float r8 = kumaraswamy_cdf(img_smoothed(i,  ym), ts_a, ts_b);
    float r9 = kumaraswamy_cdf(img_smoothed(xp, ym), ts_a, ts_b);

    float eps = 1e-7;
    float c1 = max((1.0/4)*(Dxy(xm,  j) - Dxy(i, yp))*r1, eps);
    float c2 = max((1.0/2)*(Dyy( i, yp) + Dyy(i,  j))*r2, eps);
    float c3 = max((1.0/4)*(Dxy(xp,  j) + Dxy(i, yp))*r3, eps);
    float c4 = max((1.0/2)*(Dxx(xm,  j) + Dxx(i,  j))*r4, eps);
    float c6 = max((1.0/2)*(Dxx(xp,  j) + Dxx(i,  j))*r6, eps);
    float c7 = max((1.0/4)*(Dxy(xm,  j) + Dxy(i, ym))*r7, eps);
    float c8 = max((1.0/2)*(Dyy( i, ym) + Dyy(i,  j))*r8, eps);
    float c9 = max((1.0/4)*(Dxy(xp,  j) - Dxy(i, ym))*r9, eps);

    img_dst(i,j) = (u5 + dt*(c1*u1 + c2*u2 + c3*u3 + c4*u4
			     + c6*u6 + c7*u7 + c8*u8 + c9*u9)) /
      (1 + dt*(c1 + c2 + c3 + c4 + c6 + c7 + c8 + c9));
  }

  void
  tsonad(cv::cuda::GpuMat& img_buf1,
	 cv::cuda::GpuMat& img_buf2,
	 cv::cuda::GpuMat& img_smooth,
	 cv::cuda::GpuMat& Dxx_buf,
	 cv::cuda::GpuMat& Dxy_buf,
	 cv::cuda::GpuMat& Dyy_buf,
	 cv::Ptr<cv::cuda::Filter>& gaussian_filter,
	 float dt,
	 float ts_a,
	 float ts_b,
	 float sigma,
	 float ctang,
	 size_t niters)
  /*
   * Perona, Pietro, and Jitendra Malik. 
   * "Scale-space and edge detection using anisotropic diffusion." 
   * IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 1990.
   */
  {
    size_t M  = static_cast<size_t>(img_buf1.rows);
    size_t N  = static_cast<size_t>(img_buf1.cols);
    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));
    gaussian_filter->apply(img_buf1, img_smooth);
    usdg::tsonad_diffusion_matrix<<<grid,block>>>(img_smooth,
						  Dxx_buf,
						  Dxy_buf,
						  Dyy_buf,
						  sigma,
						  ctang);
    for (size_t i = 0; i < niters; ++i)
    {
      usdg::tsonad_diffusion<<<grid,block>>>(img_buf1,
					     img_smooth,
					     Dxx_buf,
					     Dxy_buf,
					     Dyy_buf,
					     img_buf2,
					     ts_a,
					     ts_b,
					     dt);
      cv::swap(img_buf1, img_buf2);
    }
    cuda_check( cudaPeekAtLastError() );
  }

  TSONAD::
  TSONAD()
    : _img_buf1(),
    _img_buf2(),
    _img_smooth(),
    _Dxx_buf(),
    _Dxy_buf(),
    _Dyy_buf(),
    _gaussian_filter()
  {}

  void
  TSONAD::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _img_buf1.create(n_rows, n_cols, CV_32F);
    _img_buf2.create(n_rows, n_cols, CV_32F);
    _img_smooth.create(n_rows, n_cols, CV_32F);
    _Dxx_buf.create(n_rows, n_cols, CV_32F);
    _Dxy_buf.create(n_rows, n_cols, CV_32F);
    _Dyy_buf.create(n_rows, n_cols, CV_32F);
    _gaussian_filter = cv::cuda::createGaussianFilter(
      CV_32F, CV_32F, cv::Size(5,5), 4.0);
  }

  void
  TSONAD::
  apply(cv::Mat const& image,
	cv::Mat&       output,
	float dt,
	float ts_a,
	float ts_b,
	float sigma,
	float ctang,
	size_t niters)
  {
    _img_buf1.upload(image);
    usdg::tsonad(_img_buf1,
		_img_buf2,
		_img_smooth,
		_Dxx_buf,
		_Dxy_buf,
		_Dyy_buf,
		_gaussian_filter,
		 dt, ts_a, ts_b, sigma, ctang, niters);
    _img_buf2.download(output);
  }
}