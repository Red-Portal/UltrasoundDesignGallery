
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

#include "local_laplacian.hpp"

#include "cuda_utils.hpp"

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <cmath>

namespace usdg
{
  __global__ void
  local_laplacian_impl(int M, int N,
		       cv::cuda::PtrStepSzf       const G,
		       cv::cuda::PtrStepSz<uchar> const mask,
		       cv::cuda::PtrStepSzf       L,
		       float                      alpha,
		       float                      beta,
		       float                      sigma)
  {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    
    if (i >= M || j >= N || mask(i,j) == 0)
      return;

    float g     = G(i,j);
    float l     = L(i,j);
    float delta = l - g;

    if (abs(delta) <= sigma)
      L(i,j) = g + usdg::sign(delta)*sigma*pow(abs(delta) / sigma, alpha);
    else
      L(i,j) = g + usdg::sign(delta)*(beta*(abs(delta) - sigma) + sigma);
  }

  LocalLaplacian::
  LocalLaplacian()
    : _L(),
    _G(),
    _mask()
  { }

  void 
  LocalLaplacian::
  preallocate(size_t n_rows, size_t n_cols)
  {
    _L.create(   n_rows, n_cols, CV_32F);
    _G.create(   n_rows, n_cols, CV_32F);
    _mask.create(n_rows, n_cols, CV_8U);
  }

  void
  LocalLaplacian::
  apply(cv::Mat const& G,
	cv::Mat const& mask,
	cv::Mat& L,
	float alpha,
	float beta,
	float sigma_g)
  {
    size_t M  = static_cast<size_t>(G.rows);
    size_t N  = static_cast<size_t>(G.cols);
    const dim3 block(8,8);
    const dim3 grid(static_cast<unsigned int>(
		      ceil(static_cast<float>(M)/block.x)),
		    static_cast<unsigned int>(
		      ceil(static_cast<float>(N)/block.y)));

    auto roi = cv::Rect(0, 0, N, M);

    auto roi_G    = _G(roi);
    auto roi_L    = _L(roi);
    auto roi_mask = _mask(roi);

    roi_G.upload(G);
    roi_L.upload(L);
    roi_mask.upload(mask);

    usdg::local_laplacian_impl<<<grid, block>>>(
      M, N, _G, _mask, _L, alpha, beta, sigma_g);
    roi_L.download(L);
  }
}
