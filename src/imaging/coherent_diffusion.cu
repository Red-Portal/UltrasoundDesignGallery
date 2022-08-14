
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

#include "coherent_diffusion.hpp"

#include "utils.hpp"
#include "cuda_utils.hpp"

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <cmath>

namespace usdg
{
  __global__ void
  ncd_structure_tensor(int M, int N,
		       cv::cuda::PtrStepSzf       const img,
		       cv::cuda::PtrStepSz<uchar> const mask,
		       cv::cuda::PtrStepSzf             J_xx,
		       cv::cuda::PtrStepSzf             J_xy,
		       cv::cuda::PtrStepSzf             J_yy,
		       cv::cuda::PtrStepSzf             G_x,
		       cv::cuda::PtrStepSzf             G_y)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    int xp = i+1;
    int xm = i-1;
    int yp = j+1;
    int ym = j-1;

    float dudx = optimized_derivative_x(img, mask, i, j, xp, xm, yp, ym, M, N);
    float dudy = optimized_derivative_y(img, mask, i, j, xp, xm, yp, ym, M, N);

    G_x(i,j) = dudx;
    G_y(i,j) = dudy;

    J_xx(i,j) = dudx*dudx;
    J_xy(i,j) = dudx*dudy;
    J_yy(i,j) = dudy*dudy;
  }

  __global__ void
  ncd_diffusion_matrix(int M, int N,
		       cv::cuda::PtrStepSzf const J_xx_rho,
		       cv::cuda::PtrStepSzf const J_xy_rho,
		       cv::cuda::PtrStepSzf const J_yy_rho,
		       cv::cuda::PtrStepSzf const G_x,
		       cv::cuda::PtrStepSzf const G_y,
		       cv::cuda::PtrStepSz<uchar> const mask,
		       float alpha,
		       float s,
		       cv::cuda::PtrStepSzf j1,
		       cv::cuda::PtrStepSzf j2)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    float v1x, v1y, v2x, v2y, mu1, mu2;
    eigenbasis_2d(J_xx_rho(i,j),
		  J_xy_rho(i,j),
		  J_yy_rho(i,j),
		  v1x, v1y, v2x, v2y, mu1, mu2);
    float delta_mu = mu1 - mu2;
    float kappa    = delta_mu*delta_mu;
    float lambda1  = 0.0;
    float lambda2  = alpha;

    if (kappa < s*s)
      lambda1 = alpha*(1 - kappa/(s*s));

    float dudx = G_x(i,j);
    float dudy = G_y(i,j);
    float a    = lambda1*v1x*v1x + lambda2*v2x*v2x;
    float b    = lambda1*v1x*v1y + lambda2*v2x*v2y;
    float c    = lambda1*v1y*v1y + lambda2*v2y*v2y;

    j1(i,j) = a*dudx + b*dudy;
    j2(i,j) = b*dudx + c*dudy;
  }

  __global__ void
  ncd_diffusion(int M, int N,
		cv::cuda::PtrStepSzf       const img_src,
		cv::cuda::PtrStepSz<uchar> const mask,
		cv::cuda::PtrStepSzf       const j1,
		cv::cuda::PtrStepSzf       const j2,
		float dt, 
		cv::cuda::PtrStepSzf       img_dst)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    int xp = i+1;
    int xm = i-1;
    int yp = j+1;
    int ym = j-1;

    float dj1dx = usdg::optimized_derivative_x(j1, mask, i, j, xp, xm, yp, ym, M, N);
    float dj2dy = usdg::optimized_derivative_y(j2, mask, i, j, xp, xm, yp, ym, M, N);

    img_dst(i,j) = img_src(i,j) + dt*(dj1dx + dj2dy);
  }

  CoherentDiffusion::
  CoherentDiffusion()
    : _img_buf1(),
    _img_buf2(),
    _J_xx(),
    _J_xy(),
    _J_yy(),
    _J_xx_rho(),
    _J_xy_rho(),
    _J_yy_rho(),
    _G_x(),
    _G_y(),
    _j1(),
    _j2(),
    _img_in_buf(),
    _img_out_buf(),
    _mask_buf()
  {}

  void
  CoherentDiffusion::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _img_buf1.create(   n_rows, n_cols, CV_32F);
    _img_buf2.create(   n_rows, n_cols, CV_32F);
    _J_xx.create(       n_rows, n_cols, CV_32F);
    _J_xy.create(       n_rows, n_cols, CV_32F);
    _J_yy.create(       n_rows, n_cols, CV_32F);
    _J_xx_rho.create(   n_rows, n_cols, CV_32F);
    _J_xy_rho.create(   n_rows, n_cols, CV_32F);
    _J_yy_rho.create(   n_rows, n_cols, CV_32F);
    _G_x.create(        n_rows, n_cols, CV_32F);
    _G_y.create(        n_rows, n_cols, CV_32F);
    _j1.create(         n_rows, n_cols, CV_32F);
    _j2.create(         n_rows, n_cols, CV_32F);
    _img_in_buf.create( n_rows, n_cols, CV_32F);
    _img_out_buf.create(n_rows, n_cols, CV_32F);
    _mask_buf.create(   n_rows, n_cols, CV_8U);
  }

  void
  CoherentDiffusion::
  apply(cv::cuda::GpuMat const& image,
	cv::cuda::GpuMat const& mask,
	cv::cuda::GpuMat&       output,
	float rho, float alpha, float s,
	float dt, int n_iters)
  {
    _img_buf1.setTo(cv::Scalar(0.f));
    _img_buf2.setTo(cv::Scalar(0.f));

    size_t M  = static_cast<size_t>(image.rows);
    size_t N  = static_cast<size_t>(image.cols);

    image.copyTo(_img_buf1);
    image.copyTo(_img_buf2);

    auto gaussian_filter = cv::cuda::createGaussianFilter(
      CV_32F, CV_32F, cv::Size(5, 5), rho);

    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    for (size_t i = 0; i < n_iters; ++i)
    {
      usdg::ncd_structure_tensor<<<grid,block>>>(
	M, N, _img_buf1, mask, _J_xx, _J_xy, _J_yy, _G_x, _G_y);
      
      gaussian_filter->apply(_J_xx, _J_xx_rho);
      gaussian_filter->apply(_J_xy, _J_xy_rho);
      gaussian_filter->apply(_J_yy, _J_yy_rho);

      ncd_diffusion_matrix<<<grid, block>>>(
	M, N,
	_J_xx_rho, _J_xy_rho, _J_yy_rho,
	_G_x, _G_y,
	mask,
	alpha, s,
	_j1, _j2);

      usdg::ncd_diffusion<<<grid,block>>>(
	M, N,
	_img_buf1,
	mask,
	_j1, _j2,
	dt,
	_img_buf2);

      cv::swap(_img_buf1, _img_buf2);
    }
    cuda_check( cudaPeekAtLastError() );
    _img_buf1.copyTo(output);
  }

  void
  CoherentDiffusion::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	cv::Mat&       output,
	float rho, float alpha, float s,
	float dt, int n_iters)
  {
    _img_in_buf.upload(image);
    _mask_buf.upload(  mask);
    this->apply(_img_in_buf,
		_mask_buf,
		_img_out_buf,
		rho, alpha, s,
		dt, n_iters);
    _img_out_buf.download(output);
  }
}