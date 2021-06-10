
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

#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>

namespace usdg
{
  inline std::pair<cv::Mat, std::vector<cv::Mat>>
  init_pyramid(cv::Mat const& image, size_t n_scale)
  {
    double factor   = pow(2, static_cast<double>(n_scale-1));
    int cols_padded = static_cast<int>(factor*ceil(image.cols / factor));
    int rows_padded = static_cast<int>(factor*ceil(image.rows / factor));
    int col_pad     = cols_padded - image.cols;
    int row_pad     = rows_padded - image.rows;

    auto image_padded = cv::Mat(rows_padded, cols_padded, CV_32F);
    cv::copyMakeBorder(image, image_padded,
		       0, row_pad, 0, col_pad,
		       cv::BORDER_CONSTANT,
		       cv::Scalar(0));
    auto L_buf = std::vector<cv::Mat>(n_scale);
    for (size_t i = 0; i < n_scale; ++i)
    {
      double scaling = exp2(static_cast<double>(i));
      L_buf[i] = cv::Mat(static_cast<int>(image_padded.rows / scaling),
			 static_cast<int>(image_padded.cols / scaling),
			 CV_32F);
    }
    return {std::move(image_padded), std::move(L_buf)};
  }

  inline void
  analyze_pyramid(cv::Mat const& src, std::vector<cv::Mat>& L)
  { /* Highly optimized, dirty, Laplacian pyramid decomposition */
    auto G_curr     = src.clone();
    auto G_next     = src.clone();
    size_t n_scale  = L.size();
    auto next_shape = std::vector<int>(2);
    for (size_t i = 0; i < n_scale - 1; ++i)
    {
      /* setup buffer for G_{k+1} */
      next_shape[0] = L[i+1].rows;
      next_shape[1] = L[i+1].cols;
      G_next.reshape(0, next_shape);

      /* G_{k+1} = down(G_{k}) */
      cv::pyrDown(G_curr, G_next);

      /* L_k = G_k - up(L_{k+1}) */
      cv::pyrUp(G_next, L[i]);
      L[i] = G_curr - L[i];

      /* G_next <- G_curr */
      cv::swap(G_curr, G_next);
    }
    L[n_scale-1] = G_curr;
  }

  inline void
  synthesize_pyramid_inplace(std::vector<cv::Mat>& L)
  { /* Highly optimized, dirty, Laplacian pyramid decomposition */
    size_t n_scale  = L.size();
    auto L_prev     = cv::Mat(L[0].rows, L[0].cols, CV_32F);
    auto curr_shape = std::vector<int>(2);

    for (int i = static_cast<int>(n_scale)-2; i >= 0; --i)
    {
      size_t idx = static_cast<size_t>(i);
      /* setup buffer for G_{k+1} */
      curr_shape[0] = L[idx].cols;
      curr_shape[1] = L[idx].rows;
      L_prev.reshape(0, curr_shape);

      cv::pyrUp(L[idx+1], L_prev);
      L[idx] += L_prev;
    }
  }
}


#endif
