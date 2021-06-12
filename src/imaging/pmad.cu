
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
#include "utils.hpp"

namespace usdg
{
  __device__ __forceinline__ float
  pmad_coefficient(float gradient_norm, float K)
  {
    float coef = (gradient_norm / K);
    return 1/(1 + coef*coef);
  }

  __global__ void
  pmad_kernel(cv::cuda::PtrStepSzf const src,
	      cv::cuda::PtrStepSzf dst,
	      float lambda,
	      float K)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    int M = src.rows;
    int N = src.cols;

    if (i >= M || j >= N)
      return;

    float w = src(max(i-1, 0), j          );
    float n = src(i,           max(j-1, 0));
    float c = src(i,           j          );
    float s = src(i,           min(j+1, N));
    float e = src(min(i+1, M), j          );

    float g_n = n - c;
    float g_s = s - c;
    float g_w = w - c;
    float g_e = e - c;

    float Cn  = pmad_coefficient(abs(g_n), K);
    float Cs  = pmad_coefficient(abs(g_s), K);
    float Cw  = pmad_coefficient(abs(g_w), K);
    float Ce  = pmad_coefficient(abs(g_e), K);

    dst(i, j) = (c + lambda*(Cw*w + Cn*n + Ce*e + Cs*s))
      / (1 + lambda*(Cw + Cn + Ce + Cs));
  }

  void
  pmad(cv::cuda::GpuMat& buf1,
       cv::cuda::GpuMat& buf2,
       float lambda,
       float K,
       size_t niters)
  /*
   * Perona, Pietro, and Jitendra Malik. 
   * "Scale-space and edge detection using anisotropic diffusion." 
   * IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 1990.
   */
  {
    size_t M  = buf1.rows;
    size_t N  = buf1.cols;
    const dim3 block(8,8);
    const dim3 grid(ceil(static_cast<float>(M)/block.x),
		    ceil(static_cast<float>(N)/block.y));
    for (size_t i = 0; i < niters; ++i)
    {
      usdg::pmad_kernel<<<grid, block>>>(buf1, buf2, lambda, K);
      cv::swap(buf1, buf2);
    }
    cuda_check( cudaPeekAtLastError() );
  }

  PMAD::
  PMAD(size_t n_rows, size_t n_cols)
    : _device_buf1(),
      _device_buf2()
  {
    _device_buf1.create(n_rows, n_cols, CV_32F);
    _device_buf2.create(n_rows, n_cols, CV_32F);
  }

  void
  PMAD::
  apply(cv::Mat const& image,
       cv::Mat&       output,
       float lambda,
       float K,
       size_t niters)
  {
    _device_buf1.upload(image);
    pmad(_device_buf1, _device_buf2, lambda, K, niters);
    _device_buf2.download(output);
  }
}