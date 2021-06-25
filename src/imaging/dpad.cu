
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

#include "dpad.hpp"
#include "utils.hpp"

namespace usdg
{
  __global__ void
  dpad_coef_kernel1(cv::cuda::PtrStepSzf const G,
		    cv::cuda::PtrStepSzf G_coef2)
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

    float G_icov2 = (mu_G2 - mu_G*mu_G) / max(mu_G*mu_G, 1e-7);
    G_coef2(i, j) = G_icov2;
  }

  __global__ void
  dpad_coef_kernel2(cv::cuda::PtrStepSzf coef2_srcdst,
		    float noise_coef2)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = coef2_srcdst.rows;
    int N = coef2_srcdst.cols;

    if (i >= M || j >= N)
      return;

    coef2_srcdst(i,j) = (noise_coef2*(coef2_srcdst(i,j) + 1))
      / max(coef2_srcdst(i,j)*(noise_coef2 + 1), 1e-7);
  }

  __global__ void
  dpad_diffusion(cv::cuda::PtrStepSzf const src,
		 cv::cuda::PtrStepSzf const coef2,
		 cv::cuda::PtrStepSzf dst,
		 float dt)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = src.rows;
    int N = src.cols;

    if (i >= M || j >= N)
      return;

    float I_w = src(max(i-1, 0), j          );
    float I_n = src(i,           max(j-1, 0));
    float I_c = src(i,           j          );
    float I_s = src(i,           min(j+1, N));
    float I_e = src(min(i+1, M), j          );

    float C_w = coef2(max(i-1, 0), j          );
    float C_n = coef2(i,           max(j-1, 0));
    float C_c = coef2(i,           j          );
    float C_s = coef2(i,           min(j+1, N));
    float C_e = coef2(min(i+1, M), j          );
	
    dst(i, j)  = (I_c + dt*(C_w*I_w + C_n*I_n + C_e*I_e + C_s*I_s))
      / (1 + dt*(C_w + C_n + C_e + C_s));
  }

  void
  dpad(cv::cuda::GpuMat& G_buf1,
       cv::cuda::GpuMat& G_buf2,
       cv::cuda::GpuMat& G_coef2_buf,
       cv::cuda::GpuMat& G_noise_coef2_buf,
       float dt,
       size_t niters)
  /*
   * Perona, Pietro, and Jitendra Malik. 
   * "Scale-space and edge detection using anisotropic diffusion." 
   * IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 1990.
   */
  {
    size_t M  = static_cast<size_t>(G_buf1.rows);
    size_t N  = static_cast<size_t>(G_buf1.cols);
    auto buf  = cv::Mat(M, N, CV_32F);

    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));
    for (size_t i = 0; i < niters; ++i)
    {
      usdg::dpad_coef_kernel1<<<grid, block>>>(G_buf1, G_coef2_buf);
      auto mean = cv::cuda::sum(G_coef2_buf)[0] / (M*N);
      usdg::dpad_coef_kernel2<<<grid, block>>>(G_coef2_buf, mean);
      usdg::dpad_diffusion<<<grid, block>>>(G_buf1, G_coef2_buf, G_buf2, dt);
      cv::swap(G_buf1, G_buf2);
    }
    cuda_check( cudaPeekAtLastError() );
  }

  DPAD::
  DPAD()
    : _image_buf1(),
    _image_buf2(),
    _coef_buf(),
    _noise_coef_buf()
  {}

  void
  DPAD::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _image_buf1.create(n_rows, n_cols, CV_32F);
    _image_buf2.create(n_rows, n_cols, CV_32F);
    _coef_buf.create(n_rows, n_cols, CV_32F);
    _noise_coef_buf.create(n_rows, n_cols, CV_32F);
  }

  void
  DPAD::
  apply(cv::Mat const& image,
       cv::Mat&       output,
       float dt,
       size_t niters)
  {
    _image_buf1.upload(image);
    dpad(_image_buf1, _image_buf2, _coef_buf, _noise_coef_buf, dt, niters);
    _image_buf2.download(output);
  }
}