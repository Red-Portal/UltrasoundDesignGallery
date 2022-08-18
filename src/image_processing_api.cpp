
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

#include "imaging/cascaded_pyramid.hpp"

#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <iostream>

extern "C" {
  void process_image_c_api(float* img, float* mask,
			   size_t M, size_t N,
			   float* params,
			   float* output) 
  {
    auto filter    = usdg::CascadedPyramid(M, N);
    auto img_cv    = cv::Mat(M, N, CV_32F, img);
    auto mask_cv   = cv::Mat(M, N, CV_8U,  mask);
    auto output_cv = cv::Mat(M, N, CV_32F);

    float ee1_beta   = params[0];
    float ee1_sigma  = params[1];
    float ee2_beta   = params[2];
    float ee2_sigma  = params[3];
    float ncd1_s     = params[4];
    float ncd1_alpha = params[5];
    float ncd2_s     = params[6];
    float ncd2_alpha = params[7];
    float rpncd_k    = params[8];
    // filter.apply(img_cv, mask_cv, output_cv,
    // 		 ee1_beta,
    // 		 ee1_sigma,
    // 		 ee2_beta,
    // 		 ee2_sigma,
    // 		 ncd1_s,
    // 		 ncd1_alpha,
    // 		 ncd2_s,
    // 		 ncd2_alpha,
    // 		 rpncd_k);
    auto out_cv_ptr = reinterpret_cast<float*>(output_cv.data);
    std::copy(out_cv_ptr, out_cv_ptr + M*N, output);
  }
}
