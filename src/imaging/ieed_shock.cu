
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

#include "ieed_shock.hpp"

#include "utils.hpp"
#include "cuda_utils.hpp"

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <cmath>

namespace usdg
{
  __global__ void
  ieedshock_compute_structure_tensor(cv::cuda::PtrStepSzf       const img,
				     cv::cuda::PtrStepSz<uchar> const mask,
				     cv::cuda::PtrStepSzf             J_xx,
				     cv::cuda::PtrStepSzf             J_xy,
				     cv::cuda::PtrStepSzf             J_yy,
				     cv::cuda::PtrStepSzf             G_x,
				     cv::cuda::PtrStepSzf             G_y)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = img.rows;
    int N = img.cols;

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

    G_x(i,j) = g_x;
    G_y(i,j) = g_y;
  }

  __global__ void
  ieedshock_compute_diffusion_matrix(cv::cuda::PtrStepSzf const J_xx_rho,
				     cv::cuda::PtrStepSzf const J_xy_rho,
				     cv::cuda::PtrStepSzf const J_yy_rho,
				     cv::cuda::PtrStepSzf const G_x,
				     cv::cuda::PtrStepSzf const G_y,
				     cv::cuda::PtrStepSz<uchar> const mask,
				     float m1,
				     float m2,
				     float k1,
				     float k2,
				     float Cm1,
				     float Cm2,
				     cv::cuda::PtrStepSzf D_xx,
				     cv::cuda::PtrStepSzf D_xy,
				     cv::cuda::PtrStepSzf D_yy,
				     cv::cuda::PtrStepSzf edge_map,
				     cv::cuda::PtrStepSzf shock)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = J_xx_rho.rows;
    int N = J_xx_rho.cols;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    int xp = min(i+1, M-1);
    int xm = max(i-1,   0);
    int yp = min(j+1, N-1);
    int ym = max(j-1,   0);

    float G_x_c  = G_x(i, j);
    float G_y_c  = G_y(i, j);
    float G_x_xp = fetch_pixel(G_x, xp,  j, mask, G_x_c);
    float G_x_xm = fetch_pixel(G_x, xm,  j, mask, G_x_c);
    float G_x_yp = fetch_pixel(G_x,  i, yp, mask, G_x_c);
    float G_x_ym = fetch_pixel(G_x,  i, ym, mask, G_x_c);
    float G_y_yp = fetch_pixel(G_y,  i, yp, mask, G_y_c);
    float G_y_ym = fetch_pixel(G_y,  i, ym, mask, G_y_c);

    float I_xx = (G_x_xp - G_x_xm) / 2;
    float I_xy = (G_x_yp - G_x_ym) / 2;
    float I_yy = (G_y_yp - G_y_ym) / 2;

    float v1x, v1y, v2x, v2y, mu1, mu2;
    eigenbasis_2d(J_xx_rho(i,j),
		  J_xy_rho(i,j),
		  J_yy_rho(i,j),
		  v1x, v1y, v2x, v2y, mu1, mu2);
    float lambda1 = 1 - __expf(-Cm1 / max(__powf(mu1 / k1, m1), 1e-7));
    float lambda2 = 1 - __expf(-Cm2 / max(__powf(mu2 / k2, m2), 1e-7));

    edge_map(i,j) = lambda1;
    D_xx(i,j)     = lambda1*v1x*v1x + lambda2*v2x*v2x;
    D_xy(i,j)     = lambda1*v1x*v1y + lambda2*v2x*v2y;
    D_yy(i,j)     = lambda1*v1y*v1y + lambda2*v2y*v2y;
    shock(i,j)    = -sign(v1x*v1x*I_xx + v1x*v1y*I_xy + v1y*v1y*I_yy);
  }

  __global__ void
  ieedshock_diffuse(cv::cuda::PtrStepSzf       const img_src,
		    cv::cuda::PtrStepSz<uchar> const mask,
		    cv::cuda::PtrStepSzf       const edge_map,
		    cv::cuda::PtrStepSzf       const D_xx,
		    cv::cuda::PtrStepSzf       const D_xy,
		    cv::cuda::PtrStepSzf       const D_yy,
		    cv::cuda::PtrStepSzf       const shock,
		    float dt, float r,
		    cv::cuda::PtrStepSzf       img_dst)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = img_src.rows;
    int N = img_src.cols;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    float c_shock = 1 - edge_map(i,j);

    int xp = min(i+1, M-1);
    int xm = max(i-1,   0);
    int yp = min(j+1, N-1);
    int ym = max(j-1,   0);

    float I_c  = img_src(i, j);
    float I_xp = fetch_pixel(img_src, xp,  j, mask, I_c);
    float I_xm = fetch_pixel(img_src, xm,  j, mask, I_c);
    float I_yp = fetch_pixel(img_src,  i, yp, mask, I_c);
    float I_ym = fetch_pixel(img_src,  i, ym, mask, I_c);

    float g_xp = I_xp - I_c;
    float g_xm = I_c  - I_xm;
    float g_yp = I_yp - I_c;
    float g_ym = I_c  - I_ym;

    float g_x   = minmod(g_xp, g_xm);
    float g_y   = minmod(g_yp, g_ym);
    float g_mag = sqrt(g_x*g_x + g_y*g_y);

    usdg::matrix_diffuse_impl(img_src, mask,
			      D_xx, D_xy, D_yy,
			      i, j,
			      xp, xm, yp, ym,
			      M, N, dt,
			      img_dst);

    img_dst(i,j) += r*c_shock*dt*shock(i,j)*g_mag;
  }

  IEEDShock::
  IEEDShock()
    :  _mask(),
    _img_buf1(),
    _img_buf2(),
    _G_x(),
    _G_y(),
    _J_xx(),
    _J_xy(),
    _J_yy(),
    _J_xx_rho(),
    _J_xy_rho(),
    _J_yy_rho(),
    _D_xx(),
    _D_xy(),
    _D_yy(),
    _edge_map(),
    _shock(),
    _gaussian_filter()
  {}

  void
  IEEDShock::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _mask.create(n_rows, n_cols, CV_8U);
    _img_buf1.create(n_rows, n_cols, CV_32F);
    _img_buf2.create(n_rows, n_cols, CV_32F);
    _G_x.create(n_rows, n_cols, CV_32F);
    _G_y.create(n_rows, n_cols, CV_32F);
    _J_xx.create(n_rows, n_cols, CV_32F);
    _J_xy.create(n_rows, n_cols, CV_32F);
    _J_yy.create(n_rows, n_cols, CV_32F);
    _J_xx_rho.create(n_rows, n_cols, CV_32F);
    _J_xy_rho.create(n_rows, n_cols, CV_32F);
    _J_yy_rho.create(n_rows, n_cols, CV_32F);
    _D_xx.create(n_rows, n_cols, CV_32F);
    _D_xy.create(n_rows, n_cols, CV_32F);
    _D_yy.create(n_rows, n_cols, CV_32F);
    _edge_map.create(n_rows, n_cols, CV_32F);
    _shock.create(n_rows, n_cols, CV_32F);
    _gaussian_filter = cv::cuda::createGaussianFilter(
      CV_32F, CV_32F, cv::Size(5, 5), 1.0);
  }

  void
  IEEDShock::
  apply(cv::Mat const& image,
	cv::Mat const& mask,
	cv::Mat&       output,
	float m1, float m2,
	float k1, float k2,
	float Cm1, float Cm2,
	float r,
	float dt, int n_iters)
  {
    auto roi       = cv::Rect(0, 0, image.cols, image.rows);
    auto roi_buf1  = _img_buf1(roi);
    auto roi_buf2  = _img_buf2(roi);
    auto roi_mask  = _mask(roi);
    roi_buf1.upload(image);
    roi_mask.upload(mask);

    size_t M  = static_cast<size_t>(image.rows);
    size_t N  = static_cast<size_t>(image.cols);
    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    usdg::ieedshock_compute_structure_tensor<<<grid,block>>>(
      _img_buf1, _mask, _J_xx, _J_xy, _J_yy, _G_x, _G_y);
    cuda_check( cudaPeekAtLastError() );

    _gaussian_filter->apply(_J_xx, _J_xx_rho);
    _gaussian_filter->apply(_J_xy, _J_xy_rho);
    _gaussian_filter->apply(_J_yy, _J_yy_rho);

    ieedshock_compute_diffusion_matrix<<<grid, block>>>(
      _J_xx_rho, _J_xy_rho, _J_yy_rho,
      _G_x, _G_y, _mask,
      m1, m2,
      k1, k2,
      Cm1, Cm2,
      _D_xx, _D_xy, _D_yy,
      _edge_map,
      _shock);

    cuda_check( cudaPeekAtLastError() );
    for (size_t i = 0; i < n_iters; ++i)
    {
      usdg::ieedshock_diffuse<<<grid,block>>>(_img_buf1,
					      _mask,
					      _edge_map,
					      _D_xx,
					      _D_xy,
					      _D_yy,
					      _shock,
					      dt, r,
					      _img_buf2);
      cv::swap(_img_buf1, _img_buf2);
    }
    cuda_check( cudaPeekAtLastError() );
    roi_buf2.download(output);
  }
}