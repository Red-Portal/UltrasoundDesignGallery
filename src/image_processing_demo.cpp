
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "imaging/lpndsf.hpp"

int main()
{
  //auto fname     = std::string("../data/image/convex_liver.png");
  //auto fname     = std::string("../data/image/forearm_gray.png");
  auto fname     = std::string("../data/image/thyroid.png");
  auto wname     = "Demo";
  auto image     = cv::imread(fname);

  //image.convertTo(image, CV_32FC1);
  size_t M             = static_cast<size_t>(image.rows);
  size_t N             = static_cast<size_t>(image.cols);
  auto image_gray      = cv::Mat();
  auto image_gray_norm = cv::Mat(static_cast<int>(M), static_cast<int>(N), CV_32F);
  cv::cvtColor(image, image_gray, cv::COLOR_RGB2GRAY);
  cv::normalize(image_gray, image_gray_norm, 0, 1, cv::NORM_MINMAX, CV_32F);

  auto processing = usdg::LPNDSF(static_cast<size_t>(image_gray_norm.rows),
				 static_cast<size_t>(image_gray_norm.cols));

  auto start = std::chrono::steady_clock::now();

  float r0 = 0.0f;
  float r1 = 0.1f;
  float r2 = 0.1f;

  float k0 = 0.1f;
  float k1 = 0.1f;
  float k2 = 0.01f;
  float k3 = 0.01f;

  float t0 = 2.0f;
  float t1 = 5.0f;
  float t2 = 2.0f;
  float t3 = 2.0f;

  float alpha = 0.8f;
  float beta  = 1.2f;

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

  cv::namedWindow(wname);
  cv::imshow(wname, image_gray_norm);
  cv::waitKey(0);

  //cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8U);
  //cv.threshold(output, 0,255,cv.THRESH_TOZERO_INV)
  //cv::imwrite("output.png", L_buf[0]);

  auto output_8u = cv::Mat(output.rows, output.cols, CV_8UC1);
  for (int i = 0; i < output.rows; ++i) {
    for (int j = 0; j < output.cols; ++j) {
      output_8u.at<unsigned char>(i,j) = cv::saturate_cast<unsigned char>(
	ceil(output.at<float>(i,j)*255.0f));
    }
  }
  
  auto output_rgba = cv::Mat();
  cv::cvtColor(output_8u, output_rgba, cv::COLOR_GRAY2RGBA);
  cv::namedWindow(wname);
  cv::imshow(wname, output_rgba);
  cv::waitKey(0);
}
