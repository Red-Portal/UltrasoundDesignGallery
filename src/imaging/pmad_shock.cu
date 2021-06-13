
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

#include "pmad.hpp"
#include "pmad_shock.hpp"
#include "utils.hpp"

namespace usdg
{
  __device__ __forceinline__ float
  pmad_coefficient(float gradient_norm, float K)
  {
    float coef = (gradient_norm / K);
    return 1/(1 + coef*coef);
  }

  __device__ __forceinline__ int sign(float x)

  { 
    int t = x < 0 ? -1 : 0;
    return x > 0 ? 1 : t;
  }

  __device__ __forceinline__ float minmod(float x, float y)
  { 
    if(x*y > 0)
    {
      return sign(x)*min(abs(x), abs(y));
    }
    else
    {
      return 0.0;
    }
  }

  __global__ void
  pmad_eta_precompute_kernel(cv::cuda::PtrStepSzf const G,
			     cv::cuda::PtrStepSzf G_dst,
			     cv::cuda::PtrStepSzf eta_dst,
			     float dt,
			     float K)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = G.rows;
    int N = G.cols;

    if (i >= M || j >= N)
      return;

    float I_xp   = G(min(i+1, M), j          );
    float I_xm   = G(max(i-1, 0), j          );
    float I_yp   = G(i,           min(j+1, N));
    float I_ym   = G(i,           max(j-1, 0));
    float I_c    = G(i,           j          );
    float I_xpyp = G(min(i+1, M), min(j+1, N));
    float I_xmyp = G(max(i-1, 0), min(j+1, N));
    float I_xpym = G(min(i+1, M), max(j-1, 0));
    float I_xmym = G(max(i-1, 0), max(j-1, 0));

    float g_xp = I_xp - I_c;
    float g_xm = I_xm - I_c;
    float g_yp = I_yp - I_c;
    float g_ym = I_ym - I_c;

    float C_xp = pmad_coefficient(abs(g_xp), K);
    float C_xm = pmad_coefficient(abs(g_xm), K);
    float C_yp = pmad_coefficient(abs(g_yp), K);
    float C_ym = pmad_coefficient(abs(g_ym), K);

    float g_x  = (I_xp - I_xm)/2;
    float g_y  = (I_yp - I_ym)/2;
    float g_xx = g_xp + g_xm;
    float g_yy = g_yp + g_ym;
    float g_xy = ((I_xpyp - I_xpym) - (I_xmyp - I_xmym))/4;

    eta_dst(i,j) = g_xx*g_x*g_x + 2*g_xy*g_x*g_y + g_yy*g_y*g_y;

    G_dst(i, j)  = (I_c + dt*(C_xp*I_xp + C_xm*I_xm + C_yp*I_yp + C_ym*I_ym))
      / (1 + dt*(C_xp + C_xm + C_yp + C_ym));
  }

  __global__ void
  pmad_shock_kernel(cv::cuda::PtrStepSzf const L,
		    cv::cuda::PtrStepSzf const G,
		    cv::cuda::PtrStepSzf const laplacian,
		    cv::cuda::PtrStepSzf output,
		    float dt,
		    float r,
		    float K)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = L.rows;
    int N = L.cols;

    if (i >= M || j >= N)
      return;

    float L_w = L(max(i-1, 0), j          );
    float L_n = L(i,           max(j-1, 0));
    float L_c = L(i,           j          );
    float L_s = L(i,           min(j+1, N));
    float L_e = L(min(i+1, M), j          );

    float G_w = G(max(i-1, 0), j          );
    float G_e = G(min(i+1, M), j          );
    float G_s = G(i,           min(j+1, N));
    float G_n = G(i,           max(j-1, 0));
    float G_c = G(i,           j);

    float G_g_n   = G_n - G_c;
    float G_g_s   = G_s - G_c;
    float G_g_w   = G_w - G_c;
    float G_g_e   = G_e - G_c;
    float G_DIi   = minmod(G_g_n, -G_g_s);
    float G_DIj   = minmod(G_g_w, -G_g_e);
    float G_g_mag = sqrt(G_DIi*G_DIi + G_DIj*G_DIj);

    float L_g_n   = L_n - L_c;
    float L_g_s   = L_s - L_c;
    float L_g_w   = L_w - L_c;
    float L_g_e   = L_e - L_c;
    float L_DIi   = minmod(L_g_n, -L_g_s);
    float L_DIj   = minmod(L_g_w, -L_g_e);
    float L_g_mag = sqrt(L_DIi*L_DIi + L_DIj*L_DIj);

    float Cn  = pmad_coefficient(abs(L_g_n), K);
    float Cs  = pmad_coefficient(abs(L_g_s), K);
    float Cw  = pmad_coefficient(abs(L_g_w), K);
    float Ce  = pmad_coefficient(abs(L_g_e), K);

    float Ieta   = laplacian(i,j);
    float dshock = -r*(1 - pmad_coefficient(G_g_mag, K))*sign(Ieta)*L_g_mag;
    output(i, j) = (L_c + dt*(Cw*L_w + Cn*L_n + Ce*L_e + Cs*L_s + dshock))
      / (1 + dt*(Cw + Cn + Ce + Cs));
    //output(i, j) = L_c + dt*dshock;
  }
  
  void
  pmad_shock(cv::cuda::GpuMat& G_buf1,
	     cv::cuda::GpuMat& G_buf2,
	     cv::cuda::GpuMat& L_buf1,
	     cv::cuda::GpuMat& L_buf2,
	     cv::cuda::GpuMat& eta_buf1,
	     cv::cuda::GpuMat& eta_buf2,
	     cv::Ptr<cv::cuda::Filter>& eta_filter,
	     float dt,
	     float r,
	     float K,
	     size_t niters)
  /*
   * Perona, Pietro, and Jitendra Malik. 
   * "Scale-space and edge detection using anisotropic diffusion." 
   * IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 1990.
   */
  {
    size_t M  = static_cast<size_t>(L_buf1.rows);
    size_t N  = static_cast<size_t>(L_buf1.cols);
    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    for (size_t i = 0; i < niters; ++i)
    {
      usdg::pmad_eta_precompute_kernel<<<grid, block>>>(G_buf1, G_buf2, eta_buf1, dt, K);
      eta_filter->apply(eta_buf1, eta_buf2);
      usdg::pmad_shock_kernel<<<grid, block>>>(L_buf1, G_buf2, eta_buf2, L_buf2, dt, r, K);
      cv::swap(L_buf1, L_buf2);
      cv::swap(G_buf1, G_buf2);
    }
    cuda_check( cudaPeekAtLastError() );
  }

  PMADShock::
  PMADShock()
    : _G_buf1(),
    _G_buf2(),
    _L_buf1(),
    _L_buf2(),
    _eta_buf1(),
    _eta_buf2(),
    _eta_filter()
  {}

  void
  PMADShock::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _G_buf1.create(n_rows, n_cols, CV_32F);
    _G_buf2.create(n_rows, n_cols, CV_32F);
    _L_buf1.create(n_rows, n_cols, CV_32F);
    _L_buf2.create(n_rows, n_cols, CV_32F);
    _eta_buf1.create(n_rows, n_cols, CV_32F);
    _eta_buf2.create(n_rows, n_cols, CV_32F);
    _eta_filter = cv::cuda::createGaussianFilter(
      CV_32F, CV_32F, cv::Size(5,5), 1.0);
  }

  void
  PMADShock::
  apply(cv::Mat const& G,
	cv::Mat const& L,
	cv::Mat&       output,
	float dt,
	float r,
	float K,
	size_t niters)
  {
    _G_buf1.upload(G);
    _L_buf1.upload(L);
    pmad_shock(_G_buf1, _G_buf2,
	       _L_buf1, _L_buf2,
	       _eta_buf1, _eta_buf2,
	       _eta_filter,
	       dt, r, K, niters);
    _L_buf2.download(output);
  }
}