
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/ximgproc.hpp>

#include "imaging/lpndsf.hpp"

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

  // auto [image_padded, L    , G]  = usdg::init_pyramid(image_gray_norm, n_scale);
  // auto [_,            L_buf, __] = usdg::init_pyramid(image, n_scale);

  // auto diffusion0 = usdg::PMADShock(L[0].rows, L[0].cols);
  // auto diffusion1 = usdg::PMADShock(L[1].rows, L[1].cols);
  // auto diffusion2 = usdg::PMADShock(L[2].rows, L[2].cols);
  // auto diffusion3 = usdg::PMAD(L[3].rows, L[3].cols);

  // float r = 0.1;
  // usdg::analyze_pyramid(image_padded, G, L);
  // diffusion0.apply(G[0], L[0], L_buf[0], 0.1, r, 0.01, 100);
  // diffusion1.apply(G[1], L[1], L_buf[1], 0.1, r, 0.01, 100);
  // diffusion2.apply(G[2], L[2], L_buf[2], 0.1, r, 0.1,  50);
  // diffusion3.apply(      L[3], L_buf[3], 0.0,    0.01, 10);

  // float alpha = 0.9;
  // float beta  = 1.2;
  // auto LP_buf = cv::Mat(L_buf[3].rows, L_buf[3].cols, CV_32F);
  // cv::log(L_buf[3] + 1e-5, L_buf[3]);
  // //cv::log(L_buf[3], L_buf[3]);
  // cv::GaussianBlur( L_buf[3], LP_buf, cv::Size(5, 5), 1.0);
  // auto HP_buf = L_buf[3] - LP_buf;
  // L_buf[3]    = alpha*LP_buf + beta*HP_buf;
  // cv::exp(L_buf[3], L_buf[3]);

  // usdg::synthesize_pyramid_inplace(L_buf);

  //diffusion0.apply(G[0], G[0], L_buf[0], 0.03, 0.01, 0.01, 1000);

  auto processing = usdg::LPNDSF(image_gray_norm.rows,
				 image_gray_norm.cols);

  auto start = std::chrono::steady_clock::now();

  float r0 = 0.1;
  float r1 = 0.1;
  float r2 = 0.1;

  float k0 = 0.01;
  float k1 = 0.01;
  float k2 = 0.01;
  float k3 = 0.01;

  float t0 = 5.0;
  float t1 = 5.0;
  float t2 = 5.0;
  float t3 = 5.0;

  float alpha = 0.9;
  float beta  = 1.2;

  auto output = cv::Mat();
  processing.apply(image_gray_norm, output,
		   r0, k0, t0,
		   r1, k1, t1,
		   r2, k2, t2,
		   k3, t3,
		   alpha, beta);

  auto stop = std::chrono::steady_clock::now();
  auto dur  = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << dur.count() << "ms" << std::endl;

  //cv::normalize(L_buf[0], out_img, 0, 255, cv::NORM_MINMAX, CV_8U);
  cv::namedWindow(wname);
  cv::imshow(wname, image_gray_norm);
  cv::waitKey(0);

  //cv::imwrite("output.png", L_buf[0]);
  cv::namedWindow(wname);
  cv::imshow(wname, output);
  cv::waitKey(0);
}
