
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

#ifndef __US_GALLERY_LAPLACIAN_HPP__
#define __US_GALLERY_LAPLACIAN_HPP__

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

namespace usdg
{
  class LocalLaplacian
  {
    /* 
     * Local Laplacian filter
     * 
     * "Local Laplacian Filters: Edge-aware Image Processing 
     *  with a Laplacian Pyramid"
     * Paris, Sylvain et al.,
     * SIGGRAPH, 2011
     */
    
  private:
    cv::cuda::GpuMat _L;
    cv::cuda::GpuMat _G;
    cv::cuda::GpuMat _mask;

  public:
    LocalLaplacian();

    void preallocate(size_t n_rows, size_t n_cols);

    void apply(cv::Mat const& G,
	       cv::Mat const& mask,
	       cv::Mat& L,
	       float alpha,
	       float beta,
	       float sigma_g);
  };
}

#endif

