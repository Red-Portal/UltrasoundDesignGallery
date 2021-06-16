
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

#include <string>
#include <iostream>

#include <opencv4/opencv2/highgui.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

#include "video_player.hpp"
#include "utils.hpp"

namespace usdg
{
  VideoPlayer::
  VideoPlayer(blaze::DynamicVector<double> const& param_init,
	      std::string const& fpath)
    : _image_base([&fpath](){
        auto image = cv::imread(fpath);
        cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX, CV_32F);
        return image;
      }()),
      _back_buffer(),
      _front_buffer(),
      _parameter(param_init),
      _parameter_lock(),
      _imageproc_thread(),
      _terminate_thread(false),
      _image_processing(static_cast<size_t>(_image_base.rows),
			static_cast<size_t>(_image_base.cols)),
      _play_icon(),
      _pause_icon(),
      _stop_icon(),
      _loop_icon()
  {
    auto desktopMode = sf::VideoMode::getDesktopMode();
    auto width       = std::min(desktopMode.width,
				static_cast<unsigned int>(_image_base.rows));
    auto height      = std::min(desktopMode.height,
				static_cast<unsigned int>(_image_base.cols));
    auto window_size = ImVec2(static_cast<float>(width),
			      static_cast<float>(height));
    ImGui::Begin("Video");
    ImGui::SetWindowSize(window_size);
    ImGui::End();
    
    _play_icon.loadFromFile(ICON("play.png"));
    _pause_icon.loadFromFile(ICON("pause.png"));
    _stop_icon.loadFromFile(ICON("stop.png"));
    _loop_icon.loadFromFile(ICON("loop.png"));

    _imageproc_thread = std::thread([&, this]
    {
      auto parameter_local = blaze::DynamicVector<double>();
      auto output_gray     = cv::Mat(_image_base.rows, _image_base.cols, CV_32FC1);
      auto output_quant    = cv::Mat(_image_base.rows, _image_base.cols, CV_8UC1);
      auto output_rgba     = cv::Mat(_image_base.rows, _image_base.cols, CV_8UC4);
      while(!_terminate_thread.load())
      {
	_parameter_lock.lock();
	parameter_local = _parameter;
	_parameter_lock.unlock();

	_image_processing.apply(_image_base, output_gray, parameter_local);
	this->quantize(output_gray, output_quant);

	cv::cvtColor(output_quant, output_rgba, cv::COLOR_GRAY2RGBA);

	_buffer_lock.lock();
	_back_buffer.create(static_cast<unsigned int>(output_rgba.cols),
			    static_cast<unsigned int>(output_rgba.rows),
			    output_rgba.ptr());
	_buffer_lock.unlock();
      }
    });
  }

  VideoPlayer::
  ~VideoPlayer()
  {
    _terminate_thread.store(true);
    _imageproc_thread.join();
  }

  void 
  VideoPlayer::
  quantize(cv::Mat const& src,
	   cv::Mat& dst)
  {
    using uchar = unsigned char; 
    for (int i = 0; i < dst.rows; ++i) {
      for (int j = 0; j < dst.cols; ++j) {
	dst.at<uchar>(i,j) = cv::saturate_cast<uchar>(
	  round(src.at<float>(i,j)*255.0f));
      }
    }
  }

  void
  VideoPlayer::
  render()
  {
    if(ImGui::Begin("Video"))
    {
      _buffer_lock.lock();
      auto size = _back_buffer.getSize();
      if(size.x != 0 && size.y != 0)
	_front_buffer.loadFromImage(_back_buffer);
      _buffer_lock.unlock();
      ImGui::Image(_front_buffer);
    }
    ImGui::End();

    if(ImGui::Begin("Video Control"))
    {
      if (ImGui::ImageButton(_play_icon)) {
      }
      ImGui::SameLine();
      if (ImGui::ImageButton(_pause_icon)) {
      }
      ImGui::SameLine();
      if (ImGui::ImageButton(_stop_icon)) {
      }
    }
    ImGui::End();
  }

  void
  VideoPlayer::
  update_parameter(blaze::DynamicVector<double> const& param)
  {
    std::lock_guard<std::mutex> guard(_parameter_lock);
    _parameter = param;
  }
}
