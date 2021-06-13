
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

#ifndef __US_GALLERY_LPNDSF_HPP__
#define __US_GALLERY_LPNDSF_HPP__

#include <opencv4/opencv2/core/utility.hpp>

#include "pmad.hpp"
#include "pmad_shock.hpp"
#include "pyramid.hpp"

namespace usdg
{
  class LPNDSF
  {
    /* 
     *Laplacian pyramid nonlinear anisotropic diffusion 
     * with shock filtering and homomorphic filtering
     * 
     * inspired by:
     * Zhang, Fan, et al. 
     * "Multiscale nonlinear diffusion and shock filter for ultrasound image enhancement." 
     * IEEE CVPR, 2006.
     * 
     * Kang, Jinbum, Jae Young Lee, and Yangmo Yoo. 
     * "A new feature-enhanced speckle reduction method 
     *  based on multiscale analysis for ultrasound b-mode imaging." 
     * IEEE Tran. Biomed. Eng. 2015.
     */
  private: 
    usdg::PMADShock _diffusion0;
    usdg::PMADShock _diffusion1;
    usdg::PMADShock _diffusion2;
    usdg::PMAD      _diffusion3;

    std::vector<cv::Mat> _gaussian_pyramid_input;
    std::vector<cv::Mat> _laplacian_pyramid_input;
    std::vector<cv::Mat> _laplacian_pyramid_output;
  public: 
    LPNDSF(size_t n_rows, size_t n_cols);

    void apply(cv::Mat const& image,
	       cv::Mat&       output,
	       float r0, float k0, float t0,
	       float r1, float k1, float t1,
	       float r2, float k2, float t2,
	                 float k3, float t3,
	       float alpha, float beta);
  };
}

#endif
