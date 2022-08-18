
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

#include "local_laplacian_pyramid.hpp"

#include "utils.hpp"
//#include "cuda_utils.hpp"

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <iostream>

#include <cmath>

namespace usdg
{
  __global__ void
  remap_image_impl(int M, int N,
		   cv::cuda::PtrStepSzf       const img,
		   cv::cuda::PtrStepSz<uchar> const mask,
		   float g,
		   float alpha,
		   float beta,
		   float sigma_r2,
		   float I_range,
		   cv::cuda::PtrStepSzf img_remap_out)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;
    
    float pixel  = img(i,j);
    float delta  = pixel - g;
    float delta2 = delta*delta;

    img_remap_out(i,j) = delta*(alpha*exp(delta2/(-2*sigma_r2))*beta + (beta - 1)) + pixel;
  }

  __global__ void
  interpolate_laplacian_pyramids_impl(int M, int N,
				      int n_quants, 
				      cv::cuda::PtrStepSzf       const G,
				      cv::cuda::PtrStepSzf*      const L_quants,
				      cv::cuda::PtrStepSz<uchar> const mask,
				      float delta,
				      cv::cuda::PtrStepSzf L_out)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    float g     = G(i,j);
    float alpha = 0.0f;
    int q_idx   = floor(g / delta);
    int q_l_idx = 0;
    int q_h_idx = 0;

    if (q_idx < 0)
    {
      q_l_idx = 0;
      q_h_idx = 0;
      alpha   = 1.0f;
    }
    else if (q_idx + 1 > n_quants - 1)
    {
      q_l_idx = n_quants - 1;
      q_h_idx = n_quants - 1;
      alpha   = 0.0f;
    }
    else
    {
      q_l_idx = q_idx;
      q_h_idx = q_idx + 1;
      alpha   = 1 - (g - (q_idx*delta))/delta;
    }

    float l_l = L_quants[q_l_idx](i,j);
    float l_h = L_quants[q_h_idx](i,j);
    L_out(i,j) = alpha*l_l + (1 - alpha)*l_h;
  }

  void
  FastLocalLaplacianPyramid::
  remap_image(cv::cuda::GpuMat const& img,
	      cv::cuda::GpuMat const& mask,
	      float g,
	      float alpha,
	      float beta,
	      float sigma_range,
	      float I_range,
	      cv::cuda::GpuMat& img_remap_out) const
  {
    size_t M = static_cast<size_t>(img.rows);
    size_t N = static_cast<size_t>(img.cols);

    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    float sigma_r2 = sigma_range*sigma_range;
    usdg::remap_image_impl<<<grid,block>>>(
      M, N, img, mask, g, alpha, beta, sigma_r2, I_range, img_remap_out);

    cuda_check( cudaPeekAtLastError() );
  }

  void
  FastLocalLaplacianPyramid::
  interpolate_laplacian_pyramids(std::vector<usdg::LaplacianPyramid>& L_quants,
				 cv::cuda::GpuMat const& G,
				 cv::cuda::GpuMat const& mask,
				 size_t level,
				 float I_range,
				 cv::cuda::GpuMat& L_out) const
  {

    size_t M = static_cast<size_t>(G.rows);
    size_t N = static_cast<size_t>(G.cols);

    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    size_t n_quants = L_quants.size();
    auto L_l_ptrs   = std::vector<cv::cuda::PtrStepSzf>(n_quants);
    for (size_t n = 0; n < n_quants; ++n)
    {
      L_l_ptrs[n] = L_quants[n].L(level);
    }

    size_t L_l_ptrs_bytes = sizeof(cv::cuda::PtrStepSzf)*n_quants;
    cv::cuda::PtrStepSzf* L_l_ptrs_dev = nullptr;
    cudaMalloc(&L_l_ptrs_dev, L_l_ptrs_bytes);
    cudaMemcpy(L_l_ptrs_dev, L_l_ptrs.data(), L_l_ptrs_bytes, cudaMemcpyHostToDevice);
    cuda_check( cudaPeekAtLastError() );

    float delta = I_range / (n_quants - 1);
    usdg::interpolate_laplacian_pyramids_impl<<<grid,block>>>(
      M, N, n_quants, G, L_l_ptrs_dev, mask, delta, L_out);

    cuda_check( cudaPeekAtLastError() );
  }
}
