
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

#ifndef __US_GALLERY_PYRAMID_HPP__
#define __US_GALLERY_PYRAMID_HPP__

#include <vector>
#include <utility>
#include <iostream>

#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>

namespace usdg
{
  inline std::tuple<std::vector<cv::Mat>,
		    std::vector<cv::Mat>,
		    std::vector<cv::Mat>>
  init_pyramid(size_t n_rows,
	       size_t n_cols,
	       size_t n_scale)
  {
    double factor   = pow(2, static_cast<double>(n_scale-1));
    int rows_padded = static_cast<int>(
      factor*ceil(static_cast<double>(n_rows) / factor));
    int cols_padded = static_cast<int>(
      factor*ceil(static_cast<double>(n_cols) / factor));

    auto L_in  = std::vector<cv::Mat>(n_scale);
    auto G_in  = std::vector<cv::Mat>(n_scale);
    auto L_out = std::vector<cv::Mat>(n_scale);
    for (size_t i = 0; i < n_scale; ++i)
    {
      int scaling = static_cast<int>(exp2(static_cast<double>(i)));
      G_in[i]  = cv::Mat(rows_padded / scaling, cols_padded / scaling, CV_32F);
      L_in[i]  = cv::Mat(rows_padded / scaling, cols_padded / scaling, CV_32F);
      L_out[i] = cv::Mat(rows_padded / scaling, cols_padded / scaling, CV_32F);
    }
    return {std::move(G_in), std::move(L_in), std::move(L_out)};
  }

  inline void
  analyze_pyramid(std::vector<cv::Mat>& G,
		  std::vector<cv::Mat>& L)
  /* The input padded image is in G[0] */
  { 
    size_t n_scale  = L.size();
    for (size_t i = 0; i < n_scale - 1; ++i)
    {
      cv::pyrDown(G[i], G[i+1]);
      cv::pyrUp(G[i+1], L[i]);
      L[i] = G[i] - L[i];
    }
    L[n_scale-1] = G[n_scale-1];
  }

  inline void
  synthesize_pyramid(std::vector<cv::Mat>& L)
  /* The output is stored in-place in L[0] */
  { 
    size_t n_scale  = L.size();
    auto L_prev     = cv::Mat(L[0].rows, L[0].cols, CV_32F);
    auto curr_shape = std::vector<int>(2);

    for (int i = static_cast<int>(n_scale)-2; i >= 0; --i)
    {
      size_t idx = static_cast<size_t>(i);
      curr_shape[0] = L[idx].cols;
      curr_shape[1] = L[idx].rows;
      L_prev.reshape(0, curr_shape);

      cv::pyrUp(L[idx+1], L_prev);
      L[idx] += L_prev;
    }
  }
}


#endif
