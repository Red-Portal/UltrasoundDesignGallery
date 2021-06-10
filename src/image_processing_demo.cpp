
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/ximgproc.hpp>

#include "imaging/diffusion.hpp"
#include "imaging/pyramid.hpp"

int main()
{
  auto fname     = std::string("../data/image/forearm.png");
  auto wname     = "Demo";
  auto image     = cv::imread(fname);
  size_t n_scale = 4;

  //image.convertTo(image, CV_32FC1);
  size_t M             = image.rows;
  size_t N             = image.cols;
  auto image_gray      = cv::Mat();
  auto image_gray_norm = cv::Mat(M, N, CV_32F);
  cv::cvtColor(image, image_gray, cv::COLOR_RGB2GRAY);
  cv::normalize(image_gray, image_gray_norm, 0, 1, cv::NORM_MINMAX, CV_32F);


  auto [image_padded, L    ] = usdg::init_pyramid(image_gray_norm, n_scale);
  auto [_,            L_buf] = usdg::init_pyramid(image, n_scale);


  auto diffusion0 = usdg::Diffusion(L[0].rows, L[0].cols);
  auto diffusion1 = usdg::Diffusion(L[1].rows, L[1].cols);
  auto diffusion2 = usdg::Diffusion(L[2].rows, L[2].cols);
  auto diffusion3 = usdg::Diffusion(L[3].rows, L[3].cols);

  auto start = std::chrono::steady_clock::now();

  usdg::analyze_pyramid(image_padded, L);
  diffusion0.apply(L[0], L_buf[0], 0.01, 0.1, 100);
  diffusion1.apply(L[1], L_buf[1], 0.01, 0.1, 100);
  diffusion2.apply(L[2], L_buf[2], 0.01, 0.1, 100);
  diffusion3.apply(L[3], L_buf[3], 0.01, 0.1, 100);
  usdg::synthesize_pyramid_inplace(L_buf);

  auto stop = std::chrono::steady_clock::now();
  auto dur  = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << dur.count() << "ms" << std::endl;

  cv::namedWindow(wname);
  cv::imshow(wname, image_gray_norm);
  cv::waitKey(0);

  cv::namedWindow(wname);
  cv::imshow(wname, L_buf[0]);
  cv::waitKey(0);
}
