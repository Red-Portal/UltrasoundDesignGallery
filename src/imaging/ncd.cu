
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

#include "ncd.hpp"

#include "utils.hpp"
#include "cuda_utils.hpp"

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <cmath>

namespace usdg
{
  __global__ void
  ncd_compute_structure_tensor(int M, int N,
			       cv::cuda::PtrStepSzf       const img,
			       cv::cuda::PtrStepSz<uchar> const mask,
			       cv::cuda::PtrStepSzf             J_xx,
			       cv::cuda::PtrStepSzf             J_xy,
			       cv::cuda::PtrStepSzf             J_yy)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    int xp = min(i+1, M-1);
    int xm = max(i-1,   0);
    int yp = min(j+1, N-1);
    int ym = max(j-1,   0);

    float I_c  = img(i,j);
    float I_xp = fetch_pixel(img, xp,  j, mask, I_c);
    float I_xm = fetch_pixel(img, xm,  j, mask, I_c);
    float I_yp = fetch_pixel(img,  i, yp, mask, I_c);
    float I_ym = fetch_pixel(img,  i, ym, mask, I_c);

    float g_x   = (I_xp - I_xm) / 2;
    float g_y   = (I_yp - I_ym) / 2;

    J_xx(i,j)   = g_x*g_x;
    J_xy(i,j)   = g_x*g_y;
    J_yy(i,j)   = g_y*g_y;
  }

  __global__ void
  ncd_compute_diffusion_matrix(int M, int N,
			       cv::cuda::PtrStepSzf const J_xx_rho,
			       cv::cuda::PtrStepSzf const J_xy_rho,
			       cv::cuda::PtrStepSzf const J_yy_rho,
			       cv::cuda::PtrStepSz<uchar> const mask,
			       float alpha,
			       float s,
			       cv::cuda::PtrStepSzf D_xx,
			       cv::cuda::PtrStepSzf D_xy,
			       cv::cuda::PtrStepSzf D_yy)
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

    D_xx(i,j) = lambda1*v1x*v1x + lambda2*v2x*v2x;
    D_xy(i,j) = lambda1*v1x*v1y + lambda2*v2x*v2y;
    D_yy(i,j) = lambda1*v1y*v1y + lambda2*v2y*v2y;
  }

  __global__ void
  ncd_diffuse(int M, int N,
	      cv::cuda::PtrStepSzf       const img_src,
	      cv::cuda::PtrStepSz<uchar> const mask,
	      cv::cuda::PtrStepSzf       const D_xx,
	      cv::cuda::PtrStepSzf       const D_xy,
	      cv::cuda::PtrStepSzf       const D_yy,
	      float dt, 
	      cv::cuda::PtrStepSzf       img_dst)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    int xp = min(i+1, M-1);
    int xm = max(i-1,   0);
    int yp = min(j+1, N-1);
    int ym = max(j-1,   0);

    usdg::matrix_diffuse_impl(img_src, mask,
			      D_xx, D_xy, D_yy,
			      i, j,
			      xp, xm, yp, ym,
			      M, N, dt,
			      img_dst);
  }

  NCD::
  NCD()
    : _mask(),
    _img_buf1(),
    _img_buf2(),
    _J_xx(),
    _J_xy(),
    _J_yy(),
    _J_xx_rho(),
    _J_xy_rho(),
    _J_yy_rho(),
    _D_xx(),
    _D_xy(),
    _D_yy()
  {}

  void
  NCD::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _mask.create(n_rows, n_cols, CV_8U);
    _img_buf1.create(n_rows, n_cols, CV_32F);
    _img_buf2.create(n_rows, n_cols, CV_32F);
    _J_xx.create(n_rows, n_cols, CV_32F);
    _J_xy.create(n_rows, n_cols, CV_32F);
    _J_yy.create(n_rows, n_cols, CV_32F);
    _J_xx_rho.create(n_rows, n_cols, CV_32F);
    _J_xy_rho.create(n_rows, n_cols, CV_32F);
    _J_yy_rho.create(n_rows, n_cols, CV_32F);
    _D_xx.create(n_rows, n_cols, CV_32F);
    _D_xy.create(n_rows, n_cols, CV_32F);
    _D_yy.create(n_rows, n_cols, CV_32F);
  }

  void
  NCD::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	cv::Mat&       output,
	float rho, float alpha, float s,
	float dt, int n_iters)
  {
    _img_buf1.setTo(cv::Scalar(0));
    _img_buf2.setTo(cv::Scalar(0));

    size_t M  = static_cast<size_t>(image.rows);
    size_t N  = static_cast<size_t>(image.cols);

    auto roi       = cv::Rect(0, 0, N, M);
    auto roi_buf1  = _img_buf1(roi);
    auto roi_buf2  = _img_buf2(roi);
    auto roi_mask  = _mask(roi);
    roi_buf1.upload(image);
    roi_mask.upload(mask);

    auto gaussian_filter = cv::cuda::createGaussianFilter(
      CV_32F, CV_32F, cv::Size(9, 9), rho);

    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    usdg::ncd_compute_structure_tensor<<<grid,block>>>(
      M, N, _img_buf1, _mask, _J_xx, _J_xy, _J_yy);
    cuda_check( cudaPeekAtLastError() );

    gaussian_filter->apply(_J_xx, _J_xx_rho);
    gaussian_filter->apply(_J_xy, _J_xy_rho);
    gaussian_filter->apply(_J_yy, _J_yy_rho);

    ncd_compute_diffusion_matrix<<<grid, block>>>(
      M, N,
      _J_xx_rho, _J_xy_rho, _J_yy_rho,
      _mask,
      alpha, s,
      _D_xx, _D_xy, _D_yy);
    cuda_check( cudaPeekAtLastError() );

    for (size_t i = 0; i < n_iters; ++i)
    {
      usdg::ncd_diffuse<<<grid,block>>>(M, N,
					_img_buf1,
					_mask,
					_D_xx,
					_D_xy,
					_D_yy,
					dt,
					_img_buf2);
      cv::swap(_img_buf1, _img_buf2);
    }
    cuda_check( cudaPeekAtLastError() );
    roi_buf2.download(output);
  }
}