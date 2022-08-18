
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

#ifndef __US_GALLERY_LAPLACIAN_PYRAMID_HPP__
#define __US_GALLERY_LAPLACIAN_PYRAMID_HPP__

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <vector>

#include <cstddef>

namespace usdg
{
  class LaplacianPyramid
  {
  private:
    std::vector<cv::cuda::GpuMat> _L;
    std::vector<cv::cuda::GpuMat> _masks;
    cv::cuda::GpuMat _G_l;
    cv::cuda::GpuMat _G_l_next;
    cv::cuda::GpuMat _G_l_next_up;
  
  public:
    LaplacianPyramid(size_t n_scales);

    void apply(cv::cuda::GpuMat const& img,
	       cv::cuda::GpuMat const& mask,
	       float decimation_ratio,
	       float sigma);

    // void apply(cv::Mat const& img,
    // 	       float decimation_ratio,
    // 	       float sigma);

    void preallocate(size_t n_rows, size_t n_cols);

    inline cv::cuda::GpuMat&
    L(size_t idx)
    {
      return _L[idx];
    }

    inline cv::cuda::GpuMat&
    mask(size_t idx)
    {
      return _masks[idx];
    }
  };
}


#endif
