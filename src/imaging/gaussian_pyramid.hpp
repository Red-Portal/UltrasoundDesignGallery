
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

#ifndef __US_GALLERY_GAUSSIAN_PYRAMID_HPP__
#define __US_GALLERY_GAUSSIAN_PYRAMID_HPP__

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <vector>

#include <cstddef>

namespace usdg
{
  class GaussianPyramid
  {
  private:
    std::vector<cv::cuda::GpuMat> _G;
    float _decimation_ratio;
    cv::cuda::GpuMat _img_buffer;

  public:
    GaussianPyramid(size_t n_levels);

    void preallocate(size_t n_rows, size_t n_cols);

    void apply(cv::Mat const& img,
	       float dec_ratio,
	       float sigma);

    void apply(cv::cuda::GpuMat const& img,
	       float dec_ratio,
	       float sigma);

    inline cv::cuda::GpuMat&
    G(size_t idx)
    {
      return _G[idx];
    }

    inline size_t
    levels()
    {
      return _G.size();
    }
  };
}

#endif
