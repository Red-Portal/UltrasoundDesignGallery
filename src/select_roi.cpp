


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

#include <iostream>
#include <string>

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/tracking.hpp>
#include <portable-file-dialogs.h>

#include "metrics.hpp"

using namespace std::literals::string_literals;

int main()
{
  auto fname = pfd::open_file(
    "Select File"s, "../data",
    { "Image Files", "*.png *.jpg *.jpeg *.bmp *.tga *.gif *.psd *.hdr *.pic"
      // "Video Files"s, "*.mp4 *.wav", automate this by ffmpeg -demuxers
    }).result();

  auto image = cv::imread(fname[0]);
  cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX, CV_32F);

  while (true)
  {
    cv::Rect2d roi = cv::selectROI(image);
    auto radius    = sqrt(2)*std::min(roi.width, roi.height)/2;
    auto center_x  = roi.x + roi.width/2;
    auto center_y  = roi.y + roi.height/2;
    std::cout << "< ROI >\n"
	      << "center x: " << center_x  << ", center y: " << center_y << '\n'
	      << "radius  : " << radius 
	      << std::endl;

    cv::Rect2d back = cv::selectROI(image);
    std::cout << "< Background >:\n"
	      << "x:     " << back.x    << ", y:      " << back.y      << '\n'
	      << "width: " << back.width << ", height: " << back.height
	      << std::endl;
    std::cout << "CNR  = " << usdg::metrics::cnr(image, roi, back) << std::endl;

    auto image_roi = image.clone();
    cv::circle(image_roi,
	       cv::Point(static_cast<int>(center_x),
			 static_cast<int>(center_y)),
	       static_cast<int>(radius),
	       cv::Scalar(255,255));
    cv::rectangle(image_roi,
		  cv::Point(static_cast<int>(round(back.x)),
			    static_cast<int>(round(back.y))),
		  cv::Point(static_cast<int>(round(back.x+back.width)),
			    static_cast<int>(round(back.y+back.height))),
		  cv::Scalar(255,255));
    cv::imshow("Image with ROI", image_roi);
    cv::waitKey(0);

    cv::Rect2d ffsr = cv::selectROI(image);
    std::cout << "< FFSR >\n"
	      << "x:     " << ffsr.x     << ", y:      " << ffsr.y << '\n'
	      << "width: " << ffsr.width << ", height: " << ffsr.height
	      << std::endl;

    std::cout << "SSNR = " << usdg::metrics::ssnr(image, ffsr)     << std::endl;
  }
}
