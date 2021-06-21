
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

#ifndef __US_GALLERY_METRICS_HPP__
#define __US_GALLERY_METRICS_HPP__

#include <vector>
#include <cmath>

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/quality/qualityssim.hpp>

namespace usdg
{
  namespace metrics
  {
    inline cv::Scalar
    mssim(cv::Mat const& i1,
	  cv::Mat const& i2)
    /* 
     * Adapted from 
     * https://docs.opencv.org/master/dd/d3d/tutorial_gpu_basics_similarity.html 
     */
    {
      auto ssim_ptr = cv::quality::QualitySSIM::create(i1);
      return ssim_ptr->compute(i2);
    }

    inline double
    cnr(cv::Mat const& img,
	cv::Rect const& roi,
	cv::Rect const& back)
    {
      auto img_roi  = cv::Mat(roi.height,  roi.width,  CV_32F);
      auto img_back = cv::Mat(back.height, back.width, CV_32F);
      img(roi).convertTo( img_roi, CV_32F);
      img(back).convertTo(img_back, CV_32F);

      auto roi_mean    = cv::Scalar();
      auto roi_stddev  = cv::Scalar();
      auto back_mean   = cv::Scalar();
      auto back_stddev = cv::Scalar();
      cv::meanStdDev(img_roi,  roi_mean,  roi_stddev);
      cv::meanStdDev(img_back, back_mean, back_stddev);

      auto roi_var  = roi_stddev*roi_stddev;
      auto back_var = back_stddev*back_stddev;
      return abs(roi_mean[0] - back_mean[0]) / sqrt(roi_var[0] + back_var[0]);
    }

    inline double
    ssnr(cv::Mat const& img,
	 cv::Rect const& roi)
    {
      auto img_roi = cv::Mat(roi.height, roi.width, CV_32F);
      img(roi).convertTo(img_roi, CV_32F);

      double mu_roi  = cv::mean(img_roi)[0];
      double mu2_roi = cv::mean(img_roi.mul(img_roi))[0];
      double std_roi = sqrt(mu2_roi  - mu_roi*mu_roi);
      return mu_roi / std_roi;
    }

    // inline cv::Scalar
    // snr(cv::Mat const& image,
    // 	cv::Mat const& truth)
    // {
    //   auto image_f32 = cv::Mat(image.rows, image.cols, CV_32F);
    //   auto truth_f32 = cv::Mat(truth.rows, truth.cols, CV_32F);
    //   image.convertTo(image_f32, CV_32F);
    //   truth.convertTo(truth_f32, CV_32F);

    //   auto diff = cv::Mat(truth.rows, truth.cols, CV_32F);
    //   cv::subtract(image_f32, truth_f32, diff);

    //   auto image2 = image_f32.mul(image_f32);
    //   auto truth2 = truth_f32.mul(truth_f32);
    //   auto diff2  = diff.mul(diff);

    //   auto numerator   = cv::sum(image2 + truth2);
    //   auto denominator = cv::sum(diff2 + 1e-8);
    //   return 10*(log10(numerator[0]) - log10(denominator[0]));
    // }

    inline double
    qindex(cv::Mat const& image,
	   std::vector<cv::Mat> masks)
    {
      size_t n_masks = masks.size();
      auto means     = std::vector<cv::Scalar>(n_masks);
      auto stddev    = std::vector<cv::Scalar>(n_masks);
      auto image_f32 = cv::Mat(image.rows, image.cols, CV_32F);
      image.convertTo(image_f32, CV_32F);

      for (size_t i = 0; i < n_masks; ++i)
      {
	cv::meanStdDev(image_f32, means[i], stddev[i], masks[i]);
      }

      double numerator   = 0.0;
      double demonimator = 0.0;
      for (size_t i = 0; i < n_masks; ++i)
      {
	for (size_t j = 0; j < n_masks; ++j)
	{
	  if (i == j)
	    continue;
	  double delta = means[i][0] - means[j][0];
	  numerator += delta*delta;
	}
	demonimator += stddev[i][0]*stddev[i][0];
      }
      return numerator / demonimator;
    }
  }
}

#endif
