
/*
 * Copyright (C) 2021-2022 Ray Kim
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

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/tracking.hpp>

#include <boost/math/special_functions/lambert_w.hpp>

#include "imaging/cascaded_pyramid.hpp"
#include "imaging/laplacian_pyramid.hpp"
#include "imaging/local_laplacian_pyramid.hpp"
#include "imaging/coherent_diffusion.hpp"
#include "imaging/complex_diffusion.hpp"
#include "imaging/complex_shock.hpp"
#include "imaging/logcompression.hpp"
#include "metrics.hpp"

cv::Mat
test_laplacian(cv::Mat const& image, cv::Mat const& mask)
{
  size_t M  = static_cast<size_t>(image.rows);
  size_t N  = static_cast<size_t>(image.cols);

  auto pyr      = usdg::LaplacianPyramid(4);
  auto img_gpu  = cv::cuda::GpuMat();
  auto mask_gpu = cv::cuda::GpuMat();
  pyr.preallocate(M, N);
  img_gpu.upload( image);
  mask_gpu.upload(mask);

  //pyr.apply(img_gpu, 2.0, 2.0);

  auto start  = std::chrono::steady_clock::now();
  {
    pyr.apply(img_gpu, mask_gpu, 2.0, 2.0);
  }
  auto stop = std::chrono::steady_clock::now();
  auto dur  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << dur.count() << "us" << std::endl;

  auto recon   = cv::Mat();
  auto buf     = cv::Mat();
  pyr.L(3).download(recon);

  for (int i = 2; i >= 0; --i)
  {
    pyr.L(i).download(buf);
    cv::pyrUp(recon, recon);
    recon += buf;
  }
  return recon;
}

cv::Mat
test_local_laplacian(cv::Mat const& image, cv::Mat const& mask)
{
  size_t M  = static_cast<size_t>(image.rows);
  size_t N  = static_cast<size_t>(image.cols);

  auto pyr = usdg::FastLocalLaplacianPyramid(4, 15);
  pyr.preallocate(M, N);
  pyr.apply(image, mask, -1.0, 2.0, 30.0);

  auto start  = std::chrono::steady_clock::now();
  {
    pyr.apply(image, mask, -1.0, 1.0, 30.0);
  }
  auto stop = std::chrono::steady_clock::now();
  auto dur  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << dur.count() << "us" << std::endl;

  auto recon   = cv::Mat();
  auto buf     = cv::Mat();
  pyr.L(3).download(recon);

  for (int i = 2; i >= 0; --i)
  {
    pyr.L(i).download(buf);
    cv::pyrUp(recon, recon);
    recon += buf;
  }
  return recon;
}

cv::Mat
test_ncd(cv::Mat const& image, cv::Mat const& mask)
{
  size_t M  = static_cast<size_t>(image.rows);
  size_t N  = static_cast<size_t>(image.cols);

  auto res = cv::Mat(M, N, CV_32F);
  auto ncd = usdg::CoherentDiffusion();
  ncd.preallocate(M, N);
  ncd.apply(image, mask, res, 2.0, 0.1, 5, 2.0, 20);

  auto res_dev  = cv::cuda::GpuMat();
  auto img_dev  = cv::cuda::GpuMat();
  auto mask_dev = cv::cuda::GpuMat();
  img_dev.upload( image);
  mask_dev.upload(mask);

  auto start  = std::chrono::steady_clock::now();
  {
    ncd.apply(img_dev, mask_dev, res_dev, 2.0, 0.05, 5, 2.0, 10);
  }
  auto stop = std::chrono::steady_clock::now();
  auto dur  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << dur.count() << "us" << std::endl;
  
  res_dev.download(res);

  return res;
}

cv::Mat
test_rpncd(cv::Mat const& image, cv::Mat const& mask)
{
  size_t M  = static_cast<size_t>(image.rows);
  size_t N  = static_cast<size_t>(image.cols);

  auto res    = cv::Mat(M, N, CV_32F);
  auto filter = usdg::ComplexDiffusion();
  filter.preallocate(M, N);
  filter.apply(image, mask, res, 0.5, 5/180*3.141592, 0.3, 20);

  auto res_dev  = cv::cuda::GpuMat();
  auto img_dev  = cv::cuda::GpuMat();
  auto mask_dev = cv::cuda::GpuMat();
  img_dev.upload( image);
  mask_dev.upload(mask);
  filter.apply(img_dev, mask_dev, res_dev, 1.0, 5./180*3.141592, 0.3, 20);

  auto start  = std::chrono::steady_clock::now();
  {
    filter.apply(img_dev, mask_dev, res_dev, 0.1, 5./180*3.141592, 0.3, 20);
  }
  auto stop = std::chrono::steady_clock::now();
  auto dur  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << dur.count() << "us" << std::endl;
  
  res_dev.download(res);
  return res;
}

cv::Mat
test_cshock(cv::Mat const& image, cv::Mat const& mask)
{
  size_t M  = static_cast<size_t>(image.rows);
  size_t N  = static_cast<size_t>(image.cols);

  auto res    = cv::Mat(M, N, CV_32F);
  auto filter = usdg::ComplexShock();
  filter.preallocate(M, N);

  auto res_dev  = cv::cuda::GpuMat();
  auto img_dev  = cv::cuda::GpuMat();
  auto mask_dev = cv::cuda::GpuMat();
  img_dev.upload( image);
  mask_dev.upload(mask);
  //filter.apply(img_dev, mask_dev, res_dev, 1.0, 5./180*3.141592, 0.3, 20);

  auto start  = std::chrono::steady_clock::now();
  {
    filter.apply(img_dev, mask_dev, res_dev, 0.5, 0.5, 0.9, 0.01, 0.2, 20);
  }
  auto stop = std::chrono::steady_clock::now();
  auto dur  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << dur.count() << "us" << std::endl;
  
  res_dev.download(res);
  return res;
}

cv::Mat
test_cascaded_laplacian(cv::Mat const& image, cv::Mat const& mask)
{
  size_t M  = static_cast<size_t>(image.rows);
  size_t N  = static_cast<size_t>(image.cols);

  auto res    = cv::Mat(M, N, CV_32F);
  auto filter = usdg::CascadedPyramid(M, N);

  //filter.apply(image, mask, res);
  auto start  = std::chrono::steady_clock::now();
  {
    float llf_alpha  = -1.;
    float llf_beta   = 1.3;
    float llf_sigma  = 20.;
    float cshock_a   = 0.9;
    float ncd1_alpha = 0.05;
    float ncd1_s     = 1.;
    float ncd2_alpha = 0.05;
    float ncd2_s     = 5.;
    float rpncd_k    = 0.3;

    filter.apply(image, mask, res,
		 llf_alpha,
		 llf_beta,
		 llf_sigma,
		 cshock_a,
		 ncd1_alpha,
		 ncd1_s,
		 ncd2_alpha,
		 ncd2_s,
		 rpncd_k);
  }
  auto stop = std::chrono::steady_clock::now();
  auto dur  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << dur.count() << "us" << std::endl;

  return res;
}

int main()
{
  //auto image_fname = std::string("../data/selections/cardiac1/cardiac1_2.pfm");
  //auto mask_fname  = std::string("../data/selections/cardiac1/cardiac1_1.png");
  auto image_fname = std::string("../data/selections/liver1/liver2_1.pfm");
  auto mask_fname  = std::string("../data/selections/liver1/liver2_1.png");
  //auto image_fname = std::string("../data/results/liver1_clpda/processed_0.pfm");
  //auto mask_fname  = std::string("../data/results/liver1_clpda/processed_0.pfm");

  auto image = cv::imread(image_fname, cv::IMREAD_UNCHANGED);
  auto mask  = cv::imread(mask_fname,  cv::IMREAD_GRAYSCALE);

  auto wname = "Demo";
  cv::namedWindow(wname);

  //cv::resize(image, image, cv::Size(400, 300));
  //cv::resize(mask,  mask,  cv::Size(400, 300));

  image *= 255;

  auto res = test_cascaded_laplacian(image, mask);

  image /= 255;
  res   /= 255;

  cv::imshow("before",   image);
  cv::imshow("after",    res);
  cv::imshow("residual", (image - res)/2 + 0.5);

  cv::waitKey(0);
  return 0;
}
