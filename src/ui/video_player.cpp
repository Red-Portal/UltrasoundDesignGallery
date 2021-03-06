
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
      _buffer_lock(),

      _parameter(param_init),
      _parameter_lock(),
      _imageproc_thread(),
      _terminate_thread(false),

      _image_processing(static_cast<size_t>(_image_base.rows),
			static_cast<size_t>(_image_base.cols)),
      _image_processing_lock(),

      _preview_buffer(),
      _show_preview(false),

      _play_icon(),
      _pause_icon(),
      _stop_icon(),
      _loop_icon()
  {
    auto desktopMode = sf::VideoMode::getDesktopMode();
    auto width       = std::min(desktopMode.width,
				static_cast<unsigned int>(_image_base.cols));
    auto height      = std::min(desktopMode.height,
				static_cast<unsigned int>(_image_base.rows));
    auto window_size = ImVec2(static_cast<float>(width),
			      static_cast<float>(height));
    ImGui::Begin("Video");
    ImGui::SetWindowSize(window_size);
    ImGui::End();
    
    _play_icon.loadFromFile(ICON("play.png"));
    _pause_icon.loadFromFile(ICON("pause.png"));
    _stop_icon.loadFromFile(ICON("stop.png"));
    _loop_icon.loadFromFile(ICON("loop.png"));

    _back_buffer   = cv::Mat(_image_base.rows, _image_base.cols, CV_8UC4);
    _front_buffer.create(static_cast<unsigned int>(_image_base.cols),
			 static_cast<unsigned int>(_image_base.rows));
    _preview_buffer.create(static_cast<unsigned int>(_image_base.cols),
			   static_cast<unsigned int>(_image_base.rows));
    _imageproc_thread = std::thread([this]
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

	_image_processing_lock.lock();
	_image_processing.apply(_image_base, output_gray, parameter_local);
	_image_processing_lock.unlock();
	this->quantize(output_gray, output_quant);
	cv::cvtColor(output_quant, output_rgba, cv::COLOR_GRAY2RGBA);

	_buffer_lock.lock();
	std::swap(_back_buffer, output_rgba);
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
      _front_buffer.update(_back_buffer.data);
      _buffer_lock.unlock();
      ImGui::Image(_front_buffer);
    }
    ImGui::End();

    if(_show_preview)
    {
      if(ImGui::Begin("Preview Best Setting"))
      {
	ImGui::Image(_preview_buffer);
      }
      ImGui::End();
    }

    // if(ImGui::Begin("Video Control"))
    // {
    //   if (ImGui::ImageButton(_play_icon)) {
    //   }
    //   ImGui::SameLine();
    //   if (ImGui::ImageButton(_pause_icon)) {
    //   }
    //   ImGui::SameLine();
    //   if (ImGui::ImageButton(_stop_icon)) {
    //   }
    //   ImGui::End();
    // }
  }

  void
  VideoPlayer::
  update_parameter(blaze::DynamicVector<double> const& param)
  {
    std::lock_guard<std::mutex> guard(_parameter_lock);
    _parameter = param;
  }

  void
  VideoPlayer::
  update_preview(blaze::DynamicVector<double> const& param)
  {
    auto output_gray  = cv::Mat(_image_base.rows, _image_base.cols, CV_32FC1);
    auto output_quant = cv::Mat(_image_base.rows, _image_base.cols, CV_8UC1);
    auto output_rgba  = cv::Mat(_image_base.rows, _image_base.cols, CV_8UC4);
    _image_processing_lock.lock();
    _image_processing.apply(_image_base, output_gray, param);
    _image_processing_lock.unlock();
    this->quantize(output_gray, output_quant);

    cv::cvtColor(output_quant, output_rgba, cv::COLOR_GRAY2RGBA);
    _preview_buffer.update(output_rgba.ptr());
  }

  void
  VideoPlayer::
  toggle_preview()
  {
    _show_preview = !_show_preview;
  }
}
