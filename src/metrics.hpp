
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

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/quality/qualityssim.hpp>

#include <vector>
#include <cmath>

namespace usdg
{
  namespace metrics
  {
    inline cv::Scalar
    ssim(cv::Mat const& i1,
	 cv::Mat const& i2,
	 float peak_value=1.0)
    /* 
     * Adapted from 
     * https://docs.opencv.org/master/dd/d3d/tutorial_gpu_basics_similarity.html 
     */
    {
      float C1 = powf(0.01f*peak_value, 2);
      float C2 = powf(0.03f*peak_value, 2);
      /***************************** INITS **********************************/
      cv::cuda::GpuMat gI1, gI2, gs1, tmp1,tmp2;

      gI1.upload(i1);
      gI2.upload(i2);

      gI1.convertTo(tmp1, CV_MAKE_TYPE(CV_32F, gI1.channels()));
      gI2.convertTo(tmp2, CV_MAKE_TYPE(CV_32F, gI2.channels()));

      std::vector<cv::cuda::GpuMat> vI1, vI2;
      cv::cuda::split(tmp1, vI1);
      cv::cuda::split(tmp2, vI2);
      cv::Scalar mssim;

      cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(
	vI2[0].type(), -1, cv::Size(11, 11), 1.5);

      for( size_t i = 0; i < static_cast<size_t>(gI1.channels()); ++i )
      {
	cv::cuda::GpuMat I2_2, I1_2, I1_I2;

	cv::cuda::multiply(vI2[i], vI2[i], I2_2);        // I2^2
	cv::cuda::multiply(vI1[i], vI1[i], I1_2);        // I1^2
	cv::cuda::multiply(vI1[i], vI2[i], I1_I2);       // I1 * I2

	/*************************** END INITS **********************************/
	cv::cuda::GpuMat mu1, mu2;   // PRELIMINARY COMPUTING
	gauss->apply(vI1[i], mu1);
	gauss->apply(vI2[i], mu2);

	cv::cuda::GpuMat mu1_2, mu2_2, mu1_mu2;
	cv::cuda::multiply(mu1, mu1, mu1_2);
	cv::cuda::multiply(mu2, mu2, mu2_2);
	cv::cuda::multiply(mu1, mu2, mu1_mu2);

	cv::cuda::GpuMat sigma1_2, sigma2_2, sigma12;

	gauss->apply(I1_2, sigma1_2);
	cv::cuda::subtract(sigma1_2, mu1_2, sigma1_2); // sigma1_2 -= mu1_2;

	gauss->apply(I2_2, sigma2_2);
	cv::cuda::subtract(sigma2_2, mu2_2, sigma2_2); // sigma2_2 -= mu2_2;

	gauss->apply(I1_I2, sigma12);
	cv::cuda::subtract(sigma12, mu1_mu2, sigma12); // sigma12 -= mu1_mu2;

	///////////////////////////////// FORMULA ////////////////////////////////
	cv::cuda::GpuMat t1, t2, t3;

	mu1_mu2.convertTo(t1, -1, 2, C1); // t1 = 2 * mu1_mu2 + C1;
	sigma12.convertTo(t2, -1, 2, C2); // t2 = 2 * sigma12 + C2;
	cv::cuda::multiply(t1, t2, t3);        // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	cv::cuda::addWeighted(mu1_2, 1.0, mu2_2, 1.0, C1, t1);       // t1 = mu1_2 + mu2_2 + C1;
	cv::cuda::addWeighted(sigma1_2, 1.0, sigma2_2, 1.0, C2, t2); // t2 = sigma1_2 + sigma2_2 + C2;
	cv::cuda::multiply(t1, t2, t1);                              // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	cv::cuda::GpuMat ssim_map;
	cv::cuda::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

	cv::Scalar s = cv::cuda::sum(ssim_map);
	mssim.val[i] = s.val[0] / (ssim_map.rows * ssim_map.cols);

      }
      return mssim;
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

    inline double
    ssnr(cv::Mat const& img,
	 cv::Mat const& mask)
    {
      auto mean   = cv::Scalar();
      auto stddev = cv::Scalar();
      cv::meanStdDev(img, mean, stddev, mask);
      return mean[0] / (stddev[0] + 1e-5);
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

    inline double
    gcnr(cv::Mat const& image,
	 cv::Mat const& mask1,
	 cv::Mat const& mask2)
    {
      int channels[1]        = { 0 };
      float gray_range[2]    = { 0.0, 1.0 };
      float const* ranges[2] = { gray_range };
      int n_bins[1]          = { 64 };

      auto roi1_hist = cv::Mat();
      cv::calcHist(&image,
		   1,
		   channels,
		   mask1,
		   roi1_hist,
		   1,
		   n_bins,
		   ranges,
		   true,
		   false);		
      auto roi1_area = cv::countNonZero(mask1);
      roi1_hist     /= roi1_area;

      auto roi2_hist = cv::Mat();
      cv::calcHist(&image,
		   1,
		   channels,
		   mask2,
		   roi2_hist,
		   1,
		   n_bins,
		   ranges,
		   true,
		   false);		
      auto roi2_area = cv::countNonZero(mask2);
      roi2_hist     /= roi2_area;

      auto ovl = cv::compareHist(roi1_hist, roi2_hist, cv::HISTCMP_INTERSECT);
      return 1 - ovl;
    }

    // inline double
    // fom(cv::Mat const& image, cv::Mat const& gold, double sigma)
    // {
    //   auto imag_smooth = cv::Mat();
    //   auto gold_smooth = cv::Mat();
    //   cv::GaussianBlur(image, imag_smooth, cv::Size(5,5), sigma, sigma);
    //   cv::GaussianBlur(gold,  gold_smooth, cv::Size(5,5), sigma, sigma);

    //   auto imag_8u = cv::Mat(imag_smooth.rows, imag_smooth.cols, CV_8U);
    //   auto gold_8u = cv::Mat(imag_smooth.rows, imag_smooth.cols, CV_8U);
    //   for (int i = 0; i < imag_smooth.rows; ++i) {
    // 	for (int j = 0; j < imag_smooth.cols; ++j) {
    // 	  imag_8u.at<unsigned char>(i,j) = cv::saturate_cast<unsigned char>(
    // 	    round(imag_smooth.at<float>(i,j)*255.0f));
    // 	  gold_8u.at<unsigned char>(i,j) = cv::saturate_cast<unsigned char>(
    // 	    round(gold_smooth.at<float>(i,j)*255.0f));
    // 	}
    //   }

    //   auto imag_edges = cv::Mat();
    //   auto gold_edges = cv::Mat();
    //   cv::Canny(imag_8u, imag_edges, 32, 2);
    //   cv::Canny(gold_8u, gold_edges, 32, 2);

    //   cv::namedWindow("fuck");
    //   cv::imshow("fuck", gold_8u);
    //   cv::waitKey(0);

    //   cv::namedWindow("fuck");
    //   cv::imshow("fuck", imag_8u);
    //   cv::waitKey(0);

    //   cv::namedWindow("fuck");
    //   cv::imshow("fuck", imag_edges);
    //   cv::waitKey(0);

    //   cv::namedWindow("fuck");
    //   cv::imshow("fuck", gold_edges);
    //   cv::waitKey(0);

    // 	//auto labels = cv::Mat();
    // 	//cv::distance_transform(image, gold, labels, DIST_L2 );
    // }
  }
}

#endif
