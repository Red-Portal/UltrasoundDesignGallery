
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

#ifndef __US_GALLERY_PIPELINE_HPP__
#define __US_GALLERY_PIPELINE_HPP__

#include <opencv4/opencv2/core/utility.hpp>

#include "tsonad.hpp"

namespace usdg
{
  class Pipeline
  {
  private: 
    usdg::TSONAD _diffusion;
    cv::Mat      _despeckle_buf;
    cv::Mat      _logimage_buf;
    cv::Mat      _expimage_buf;
    cv::Mat      _lopass_buf;
    cv::Mat      _hipass_buf;
    
  public: 
    Pipeline(size_t n_rows, size_t n_cols);

    void apply(cv::Mat const& image,
	       cv::Mat&       output,
	       float t,     float ts_a, float ts_b, float sigma_g,
	       float ctang, float theta, float alpha, float beta);
  };
}

#endif
