
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

#ifndef __US_GALLERY_LOCAL_LAPLACIAN_HPP__
#define __US_GALLERY_LOCAL_LAPLACIAN_HPP__

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

#include <vector>

#include <cstddef>

namespace usdg
{
  class LaplacianPyramid
  {
  private:
    std::vector<cv::Mat> _L;
    std::vector<cv::Mat> _G;
    std::vector<cv::Mat> _masks;
    cv::Mat _blur_buffer;
  
  public:
    LaplacianPyramid(size_t n_scales);

    void apply(cv::Mat const& img,
	       cv::Mat const& mask,
	       float decimation_ratio,
	       float sigma);

    inline cv::Mat&
    L(size_t idx)
    {
      return _L[idx];
    }

    inline cv::Mat&
    G(size_t idx)
    {
      return _G[idx];
    }

    inline cv::Mat&
    mask(size_t idx)
    {
      return _masks[idx];
    }
  };
}


#endif
