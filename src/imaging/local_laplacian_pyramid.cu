
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

#include <cmath>

namespace usdg
{
  __global__ void
  pixelwise_fourier_series(int M, int N,
			   float omega,
			   cv::cuda::PtrStepSzf       const img,
			   cv::cuda::PtrStepSz<uchar> const mask,
			   cv::cuda::PtrStepSzf             img_cos,
			   cv::cuda::PtrStepSzf             img_sin)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    float pixel = img(i,j);
    
    img_cos(i,j) = cos(omega*pixel);
    img_sin(i,j) = sin(omega*pixel);
  }

  void
  GaussianLocalLaplacianPyramid::
  compute_fourier_series(cv::cuda::GpuMat const& img_in,
			 cv::cuda::GpuMat const& mask,
			 float omega,
			 int T,
			 cv::cuda::GpuMat& img_cos_out,
			 cv::cuda::GpuMat& img_sin_out) const
  {
    size_t M         = static_cast<size_t>(img_in.rows);
    size_t N         = static_cast<size_t>(img_in.cols);

    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    usdg::pixelwise_fourier_series<<<grid,block>>>(
      M, N, omega, img_in, mask, img_cos_out, img_sin_out);

    cuda_check( cudaPeekAtLastError() );
  }

  __global__ void
  pixelwise_fourier_series(int M, int N,
			   float alpha_tilde,
			   float omega,
			   float m,
			   cv::cuda::PtrStepSzf       const G,
			   cv::cuda::PtrStepSzf       const G_cos,
			   cv::cuda::PtrStepSzf       const G_sin,
			   cv::cuda::PtrStepSzf       const G_cos_up,
			   cv::cuda::PtrStepSzf       const G_sin_up,
			   cv::cuda::PtrStepSz<uchar> const mask,
			   cv::cuda::PtrStepSzf       L_fourier_recon)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    float pixel = G(i,j);
    float cosG  = cos(pixel*omega);
    float sinG  = sin(pixel*omega);

    float recon = alpha_tilde*(
      (cosG*G_sin(i,j) - sinG*G_cos(i,j))
      - (cosG*G_sin_up(i,j) - sinG*G_cos_up(i,j)));

    L_fourier_recon(i,j) += m*recon;
  }

  void
  GaussianLocalLaplacianPyramid::
  fourier_recon_accumulate(float alpha_tilde,
			   float omega,
			   float m,
			   cv::cuda::GpuMat const& G,
			   cv::cuda::GpuMat const& G_cos,
			   cv::cuda::GpuMat const& G_sin,
			   cv::cuda::GpuMat const& G_cos_up,
			   cv::cuda::GpuMat const& G_sin_up,
			   cv::cuda::GpuMat const& mask,
			   cv::cuda::GpuMat& L_fourier_recon) const
  {
    size_t M         = static_cast<size_t>(G.rows);
    size_t N         = static_cast<size_t>(G.cols);

    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    usdg::pixelwise_fourier_series<<<grid,block>>>(
      M, N, alpha_tilde, omega, m,
      G, G_cos, G_sin, G_cos_up, G_sin_up, mask,
      L_fourier_recon);

    cuda_check( cudaPeekAtLastError() );
  }

  __global__ void
  pixelwise_firstlayer_fourier_series(int M, int N,
				      float alpha_tilde,
				      float omega,
				      float m,
				      cv::cuda::PtrStepSzf       const G,
				      cv::cuda::PtrStepSzf       const G_cos,
				      cv::cuda::PtrStepSzf       const G_sin,
				      cv::cuda::PtrStepSz<uchar> const mask,
				      cv::cuda::PtrStepSzf       L_fourier_recon)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    float pixel = G(i,j);
    float cosG  = cos(pixel*omega);
    float sinG  = sin(pixel*omega);

    float recon = alpha_tilde*(sinG*G_cos(i,j) - cosG*G_sin(i,j));

    L_fourier_recon(i,j) += m*recon;
  }

  void
  GaussianLocalLaplacianPyramid::
  fourier_firstlayer_accumulate(float alpha_tilde,
				float omega,
				float m,
				cv::cuda::GpuMat const& G,
				cv::cuda::GpuMat const& G_cos,
				cv::cuda::GpuMat const& G_sin,
				cv::cuda::GpuMat const& mask,
				cv::cuda::GpuMat& L_fourier_recon) const
  {
    size_t M         = static_cast<size_t>(G.rows);
    size_t N         = static_cast<size_t>(G.cols);

    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    usdg::pixelwise_firstlayer_fourier_series<<<grid,block>>>(
      M, N, alpha_tilde, omega, m,
      G, G_cos, G_sin, mask,
      L_fourier_recon);

    cuda_check( cudaPeekAtLastError() );
  }
}
