
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

#include "pyramid.hpp"
#include "ncd.hpp"
#include "rpncd.hpp"
#include "edge_enhance.hpp"

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>

namespace usdg
{
  class Pipeline
  {
  private: 
    usdg::LaplacianPyramid _pyramid;
    usdg::NCD              _ncd;
    usdg::RPNCD            _rpncd;
    usdg::EdgeEnhance      _edge_enhance;

  public: 
    Pipeline(size_t n_rows, size_t n_cols);

    void apply(cv::Mat const& image,
	       cv::Mat const& mask,
	       cv::Mat&       output,
	       float ee1_beta,
	       float ee1_sigma,
	       float ee2_beta,
	       float ee2_sigma,
	       float ncd1_s,
	       float ncd1_alpha,
	       float ncd2_s,
	       float ncd2_alpha,
	       float rpncd_k);
  };
}

#endif
