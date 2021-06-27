
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
#include <opencv4/opencv2/cudaarithm.hpp>

#include "dpad_shock.hpp"
#include "utils.hpp"
#include "cuda_utils.hpp"

namespace usdg
{
  __global__ void
  dpad_shock_median_filter(cv::cuda::PtrStepSzf const src,
			   cv::cuda::PtrStepSzf dst)
  {
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    int const j = blockIdx.y * blockDim.y + threadIdx.y;

    int M = src.rows;
    int N = src.cols;

    if(i >= M || j >= N)
      return;

    float window[9];
    window[0] = src(max(i-1, 0), max(j-1, 0));
    window[1] = src(i,           max(j-1, 0));
    window[2] = src(min(i+1, M), max(j-1, 0));
    window[3] = src(max(i-1, 0), j);
    window[4] = src(i,           j);
    window[5] = src(min(i+1, M), j);
    window[6] = src(max(i-1, 0), min(j+1, N));
    window[7] = src(i,           min(j+1, N));
    window[8] = src(min(i+1, M), min(j+1, N));

    float tmp;
    for (int c = 0 ; c < 9 - 1; c++)
    {
      for (int d = 0 ; d < 9 - c - 1; d++)
      {
	if (window[d] > window[d+1])
	{
	  tmp         = window[d];
	  window[d]   = window[d+1];
	  window[d+1] = tmp;
	}
      }
    }
    dst(i,j) = window[4];
  }

  __global__ void
  dpad_shock_coef_kernel1(cv::cuda::PtrStepSzf const G,
			  cv::cuda::PtrStepSzf const L,
			  cv::cuda::PtrStepSzf G_coef2,
			  cv::cuda::PtrStepSzf L_coef2)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = G.rows;
    int N = G.cols;

    if (i >= M || j >= N)
      return;

    float G0 = G(max(i-1, 0), max(j-1, 0));
    float G1 = G(i,           max(j-1, 0));
    float G2 = G(min(i+1, M), max(j-1, 0));
    float G3 = G(max(i-1, 0), j);
    float G4 = G(i,           j);
    float G5 = G(min(i+1, M), j);
    float G6 = G(max(i-1, 0), min(j+1, N));
    float G7 = G(i,           min(j+1, N));
    float G8 = G(min(i+1, M), min(j+1, N));

    float mu_G  = (G0 + G1 + G2
		   + G3 + G4 + G5
		   + G6 + G7 + G8) / 9;
    float mu_G2 = (G0*G0 + G1*G1 + G2*G2
		   + G3*G3 + G4*G4 + G5*G5
		   + G6*G6 + G7*G7 + G8*G8) / 9;

    float G_icov2 = (mu_G2 - mu_G*mu_G) / max(mu_G*mu_G, 1e-6);
    G_coef2(i, j) = G_icov2;

    float L0 = L(max(i-1, 0), max(j-1, 0));
    float L1 = L(i,           max(j-1, 0));
    float L2 = L(min(i+1, M), max(j-1, 0));
    float L3 = L(max(i-1, 0), j);
    float L4 = L(i,           j);
    float L5 = L(min(i+1, M), j);
    float L6 = L(max(i-1, 0), min(j+1, N));
    float L7 = L(i,           min(j+1, N));
    float L8 = L(min(i+1, M), min(j+1, N));

    float mu_L  = (L0 + L1 + L2
		   + L3 + L4 + L5
		   + L6 + L7 + L8) / 9;
    float mu_L2 = (L0*L0 + L1*L1 + L2*L2
		   + L3*L3 + L4*L4 + L5*L5
		   + L6*L6 + L7*L7 + L8*L8) / 9;

    float L_icov2 = (mu_L2 - mu_L*mu_L) / max(mu_L*mu_L, 1e-7);
    L_coef2(i, j) = L_icov2;
  }

  __global__ void
  dpad_shock_coef_kernel2(cv::cuda::PtrStepSzf L_coef2_srcdst,
			  cv::cuda::PtrStepSzf G_coef2_srcdst,
			  float L_coef2_noise,
			  float G_coef2_noise)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = L_coef2_srcdst.rows;
    int N = L_coef2_srcdst.cols;

    if (i >= M || j >= N)
      return;

    L_coef2_srcdst(i,j) = (L_coef2_noise*(L_coef2_srcdst(i,j) + 1))
      / max(L_coef2_srcdst(i,j)*(L_coef2_noise + 1), 1e-7);
    G_coef2_srcdst(i,j) = (G_coef2_noise*(G_coef2_srcdst(i,j) + 1))
      / max(G_coef2_srcdst(i,j)*(G_coef2_noise + 1), 1e-7);
  }

  __global__ void
  dpad_shock_diffusion(cv::cuda::PtrStepSzf const G_src,
		       cv::cuda::PtrStepSzf const G_coef2,
		       cv::cuda::PtrStepSzf G_dst,
		       float dt)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = G_src.rows;
    int N = G_src.cols;

    if (i >= M || j >= N)
      return;

    float G_w = G_src(max(i-1, 0), j          );
    float G_n = G_src(i,           max(j-1, 0));
    float G_c = G_src(i,           j          );
    float G_s = G_src(i,           min(j+1, N));
    float G_e = G_src(min(i+1, M), j          );

    float g_n = G_n - G_c;
    float g_s = G_s - G_c;
    float g_w = G_w - G_c;
    float g_e = G_e - G_c;

    float C_w = G_coef2(max(i-1, 0), j          );
    float C_n = G_coef2(i,           max(j-1, 0));
    float C_s = G_coef2(i,           min(j+1, N));
    float C_e = G_coef2(min(i+1, M), j          );
	
    G_dst(i, j)  = (G_c + dt*(C_w*G_w + C_n*G_n + C_e*G_e+ C_s*G_s))
      / (1 + dt*(C_w + C_n + C_e + C_s));
  }

  __global__ void
  shock_eta(cv::cuda::PtrStepSzf const G_src,
	    cv::cuda::PtrStepSzf eta_dst)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = G_src.rows;
    int N = G_src.cols;

    if (i >= M || j >= N)
      return;

    float I_xp   = G_src(min(i+1, M), j          );
    float I_xm   = G_src(max(i-1, 0), j          );
    float I_c    = G_src(i,           j          );
    float I_yp   = G_src(i,           min(j+1, N));
    float I_ym   = G_src(i,           max(j-1, 0));
    float I_xpyp = G_src(min(i+1, M), min(j+1, N));
    float I_xmyp = G_src(max(i-1, 0), min(j+1, N));
    float I_xpym = G_src(min(i+1, M), max(j-1, 0));
    float I_xmym = G_src(max(i-1, 0), max(j-1, 0));

    float g_x  = (I_xp - I_xm)/2;
    float g_y  = (I_yp - I_ym)/2;
    float g_xx = I_xp + I_xm - 2*I_c;
    float g_yy = I_yp + I_ym - 2*I_c;
    float g_xy = ((I_xpyp - I_xpym) - (I_xmyp - I_xmym))/4;

    eta_dst(i,j) = (g_xx*g_x*g_x + 2*g_xy*g_x*g_y + g_yy*g_y*g_y);
  }

  __global__ void
  dpad_shock_diffusion_shock(cv::cuda::PtrStepSzf const L_src,
			     cv::cuda::PtrStepSzf const coef2_src,
			     cv::cuda::PtrStepSzf const eta_src,
			     cv::cuda::PtrStepSzf L_dst,
			     float r,
			     float dt)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = L_src.rows;
    int N = L_src.cols;

    if (i >= M || j >= N)
      return;

    float L_w = L_src(max(i-1, 0), j          );
    float L_n = L_src(i,           max(j-1, 0));
    float L_c = L_src(i,           j          );
    float L_s = L_src(i,           min(j+1, N));
    float L_e = L_src(min(i+1, M), j          );

    float L_g_n   = L_n - L_c;
    float L_g_s   = L_s - L_c;
    float L_g_w   = L_w - L_c;
    float L_g_e   = L_e - L_c;
    float L_DIi   = minmod(L_g_n, -L_g_s);
    float L_DIj   = minmod(L_g_w, -L_g_e);
    float L_g_mag = sqrt(L_DIi*L_DIi + L_DIj*L_DIj);

    float C_w = coef2_src(max(i-1, 0), j);
    float C_n = coef2_src(i,           max(j-1, 0));
    float C_c = coef2_src(i,           j);
    float C_s = coef2_src(i,           min(j+1, N));
    float C_e = coef2_src(min(i+1, M), j);

    float Ieta   = eta_src(i,j);
    float dshock = -r*(1 - C_c)*sign(Ieta)*L_g_mag;
    L_dst(i, j) = (L_c + dt*(C_w*L_w + C_n*L_n + C_e*L_e + C_s*L_s + dshock))
      / (1 + dt*(C_w + C_n + C_e + C_s));
  }
  
  void
  dpad_shock(cv::cuda::GpuMat& G_buf1,
	     cv::cuda::GpuMat& G_buf2,
	     cv::cuda::GpuMat& L_buf1,
	     cv::cuda::GpuMat& L_buf2,
	     cv::cuda::GpuMat& L_coef2_buf,
	     cv::cuda::GpuMat& G_coef2_buf,
	     cv::cuda::GpuMat& eta_buf,
	     cv::Ptr<cv::cuda::Filter> const& eta_filter,
	     float dt,
	     float r,
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
      usdg::dpad_shock_coef_kernel1<<<grid, block>>>(G_buf1,
						     L_buf1,
						     G_coef2_buf,
						     L_coef2_buf);
      auto G_mean = cv::cuda::sum(G_coef2_buf)[0] / (M*N);
      auto L_mean = cv::cuda::sum(G_coef2_buf)[0] / (M*N);
      usdg::dpad_shock_coef_kernel2<<<grid, block>>>(G_coef2_buf,
						     L_coef2_buf,
						     G_mean,
						     L_mean);
      usdg::dpad_shock_diffusion<<<grid, block>>>(G_buf1,
						  G_coef2_buf,
						  G_buf2,
						  dt);
      auto& eta_buf_in = G_coef2_buf;
      usdg::shock_eta<<<grid, block>>>(G_buf2, eta_buf_in);
      eta_filter->apply(eta_buf_in, eta_buf);
      usdg::dpad_shock_diffusion_shock<<<grid, block>>>(L_buf1,
							L_coef2_buf,
							eta_buf,
							L_buf2,
							r,
							dt);
      cv::swap(L_buf1, L_buf2);
      cv::swap(G_buf1, G_buf2);
    } 
    cuda_check( cudaPeekAtLastError() );
  }

  DPADShock::
  DPADShock()
    : _G_buf1(),
    _G_buf2(),
    _L_buf1(),
    _L_buf2(),
    _L_coef2_buf(),
    _G_coef2_buf(),
    _eta_buf(),
    _eta_filter()
  {}

  void
  DPADShock::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _G_buf1.create(n_rows, n_cols, CV_32F);
    _G_buf2.create(n_rows, n_cols, CV_32F);
    _L_buf1.create(n_rows, n_cols, CV_32F);
    _L_buf2.create(n_rows, n_cols, CV_32F);
    _L_coef2_buf.create(n_rows, n_cols, CV_32F);
    _G_coef2_buf.create(n_rows, n_cols, CV_32F);
    _eta_buf.create(n_rows, n_cols, CV_32F);
    _eta_filter = cv::cuda::createGaussianFilter(
      CV_32F, CV_32F, cv::Size(5,5), 1.0);
  }

  void
  DPADShock::
  apply(cv::Mat const& G,
	cv::Mat const& L,
	cv::Mat&       output,
	float dt,
	float r,
	size_t niters)
  {
    _G_buf1.upload(G);
    _L_buf1.upload(L);
    dpad_shock(_G_buf1, _G_buf2,
	       _L_buf1, _L_buf2,
	       _L_coef2_buf,       _G_coef2_buf,
	       _eta_buf,
	       _eta_filter,
	       dt,
	       r,
	       niters);
    _L_buf2.download(output);
    //_G_coef2_buf.download(output);
  }
}