
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


#include "video_player.hpp"

#include "utils.hpp"

#include <opencv4/opencv2/highgui.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

#include <ranges>
#include <string>

#include <cmath>

namespace usdg
{
  void
  VideoPlayer::
  dynamic_range_adjustment(cv::Mat& src, cv::Mat const& mask,
			   float dynamic_range, float reject) const
  {
    float max_intensity = std::min(reject + dynamic_range, 1.f);
    dynamic_range       = max_intensity - reject;
    for (int i = 0; i < src.rows; ++i) {
      for (int j = 0; j < src.cols; ++j) {
	if (mask.at<uchar>(i, j) > 0)
	{
	  auto pixel = src.at<float>(i, j);
	  if (pixel < reject)
	  {
	    src.at<float>(i, j) = 0.0f;
	  }
	  else if (pixel > max_intensity)
	  {
	    src.at<float>(i, j) = 1.0f;
	  }
	  else
	  {
	    src.at<float>(i, j) = (pixel - reject) / dynamic_range;
	  }
	}
	else
	{
	  src.at<float>(i, j) = 0;
	}
      }
    }
  }

  void 
  VideoPlayer::
  frame_averaging(std::vector<cv::Mat> const& src, cv::Mat& dst,
		  size_t frame_idx, size_t n_average) const
  {
    if (src.size() == 1)
    {
      src[frame_idx].copyTo(dst);
    }
    else
    {
      auto n_avg_fwd  = static_cast<int>(
	ceil((static_cast<double>(n_average) - 1) / 2));
      auto n_avg_bwd  = static_cast<int>(
	floor((static_cast<double>(n_average) - 1) / 2));
      auto begin_idx = std::max(static_cast<int>(frame_idx) - n_avg_bwd, 0);
      auto end_idx   = std::min(static_cast<int>(frame_idx) + n_avg_fwd,
				static_cast<int>(src.size()) - 1);
      auto buffer    = cv::Mat(src[0].rows, src[0].cols, CV_32F);

      dst.setTo(cv::Scalar(0));
      auto n_average_actual = end_idx - begin_idx + 1;
      for (auto i = begin_idx; i <= end_idx; ++i)
      {
	src[static_cast<size_t>(i)].copyTo(buffer);
	dst += buffer / static_cast<double>(n_average_actual);
      }
    }
  }

  std::vector<cv::Mat>
  VideoPlayer::
  load_video(std::vector<std::string> const& paths) const
  {
    auto envelopes_view = std::ranges::ref_view(paths)
      | std::ranges::views::transform([](auto const& path){
	return cv::imread(path, cv::IMREAD_UNCHANGED);
      });
    return std::vector<cv::Mat>(envelopes_view.begin(), envelopes_view.end());
  }

  VideoPlayer::
  VideoPlayer(blaze::DynamicVector<double> const& param_init,
	      std::vector<std::string>     const& envelopes_path,
	      std::string                  const& mask_path)
    : _dynamic_range(0.8f),
      _reject(0.1f),
      _envelopes(this->load_video(envelopes_path)),
      _mask(cv::imread(mask_path,  cv::IMREAD_GRAYSCALE)),

      _frame_rate(30),
      _frame_index(0),
      _n_average(2),
      _play_video(false),

      _render_buffer(),
      _back_buffer(),
      _front_buffer(),
      _front_sprite(),
      _buffer_lock(),

      _parameter(param_init),
      _parameter_lock(),
      _imageproc_thread(),
      _terminate_thread(false),

      _image_processing(static_cast<size_t>(_envelopes[0].rows),
			static_cast<size_t>(_envelopes[0].cols)),
      _image_processing_lock(),

      _preview_buffer(),
      _preview_sprite(),
      _show_preview(false),

      _play_icon(),
      _pause_icon(),
      _prev_icon(),
      _next_icon()
  {
    auto desktopMode = sf::VideoMode::getDesktopMode();
    int env_cols     = _envelopes[0].cols;
    int env_rows     = _envelopes[0].rows;
    auto width       = std::min(desktopMode.width,  static_cast<unsigned int>(env_cols));
    auto height      = std::min(desktopMode.height, static_cast<unsigned int>(env_rows));
    auto window_size = ImVec2(static_cast<float>(width),
			      static_cast<float>(height));
    ImGui::Begin("Video");
    ImGui::SetWindowSize(window_size);
    ImGui::End();
    
    _play_icon.loadFromFile(ICON("play.png"));
    _pause_icon.loadFromFile(ICON("pause.png"));
    _next_icon.loadFromFile(ICON("next.png"));
    _prev_icon.loadFromFile(ICON("prev.png"));

    _back_buffer   = cv::Mat(env_rows, env_cols, CV_8UC4);
    _front_buffer.create(static_cast<unsigned int>(env_cols),
			 static_cast<unsigned int>(env_rows));
    _preview_buffer.create(static_cast<unsigned int>(env_cols),
			   static_cast<unsigned int>(env_rows));
    _imageproc_thread = std::thread([this, env_rows, env_cols]
    {
      auto parameter_local = blaze::DynamicVector<double>();
      auto log_image       = cv::Mat(env_rows, env_cols, CV_32F);
      auto output_gray     = cv::Mat(env_rows, env_cols, CV_32F);
      auto output_quant    = cv::Mat(env_rows, env_cols, CV_8U);
      auto output_rgba     = cv::Mat(env_rows, env_cols, CV_8UC4);

      auto frame_start_time = std::chrono::steady_clock::now();
      while(!_terminate_thread.load())
      {
	_parameter_lock.lock();
	parameter_local = _parameter;
	_parameter_lock.unlock();

	this->frame_averaging(_envelopes, log_image, _frame_index, _n_average.load());

	_image_processing_lock.lock();
	_image_processing.apply(log_image, _mask, output_gray, parameter_local);
	_image_processing_lock.unlock();

	output_gray /= _dynamic_range.load();
	this->dynamic_range_adjustment(output_gray, _mask,
				       _dynamic_range.load(),
				       _reject.load());
	this->quantize(output_gray, output_quant);
	cv::cvtColor(output_quant, output_rgba, cv::COLOR_GRAY2RGBA);


	_buffer_lock.lock();
	std::swap(_back_buffer, output_rgba);
	_buffer_lock.unlock();

	auto current_time   = std::chrono::steady_clock::now();
	auto frame_interval = std::chrono::milliseconds(
	  static_cast<int>(round(1000.0 / _frame_rate.load())));
	if (_play_video.load() &&
	    current_time - frame_start_time > frame_interval)
	{
	  auto frame_index_local = _frame_index.load();
	  frame_index_local = (frame_index_local + 1) % _envelopes.size();
	  _frame_index.store(frame_index_local);
	  frame_start_time = current_time;
	}
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
	   cv::Mat& dst) const
  {
    using uchar = unsigned char; 
    for (int i = 0; i < dst.rows; ++i) {
      for (int j = 0; j < dst.cols; ++j) {
	dst.at<uchar>(i,j) = cv::saturate_cast<uchar>(
	  floor(src.at<float>(i,j)*255.0f));
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
      auto buffer_size =  _front_buffer.getSize();
      auto window_size = ImGui::GetWindowSize();
      auto x_scale     = window_size.x / static_cast<float>(buffer_size.x);
      auto y_scale     = window_size.y / static_cast<float>(buffer_size.y);
      auto total_scale = std::min(x_scale, y_scale);

      _front_buffer.setSmooth(true);
      _front_sprite.setTexture(_front_buffer);
      _front_sprite.setScale(total_scale, total_scale);
      ImGui::Image(_front_sprite);
    }
    ImGui::End();

    if(_show_preview)
    {
      if(ImGui::Begin("Preview Best Setting"))
      {
	auto buffer_size =  _preview_buffer.getSize();
	auto window_size = ImGui::GetWindowSize();
	auto x_scale     = window_size.x / static_cast<float>(buffer_size.x);
	auto y_scale     = window_size.y / static_cast<float>(buffer_size.y);
	auto total_scale = std::min(x_scale, y_scale);

	_preview_buffer.setSmooth(true);
	_preview_sprite.setTexture(_preview_buffer);
	_preview_sprite.setScale(total_scale, total_scale);
	ImGui::Image(_preview_sprite);
      }
      ImGui::End();
    }

    if(ImGui::Begin("Video Control"))
    {
      int n_average_local = static_cast<int>(_n_average.load());
      ImGui::SliderInt("frame averaging", &n_average_local, 1, 8);
      _n_average.store(static_cast<size_t>(n_average_local));

      float dynamic_range_local = _dynamic_range.load();
      ImGui::SliderFloat("dynamic range", &dynamic_range_local, 0.5f, 1.0f);
      _dynamic_range.store(dynamic_range_local);

      float reject_local = _reject.load();
      ImGui::SliderFloat("reject level", &reject_local, 0.01f, 0.5f);
      _reject.store(reject_local);

      int frame_rate_local = static_cast<int>(_frame_rate.load());
      ImGui::SliderInt("frame rate (fps)", &frame_rate_local, 1, 40);
      _frame_rate.store(static_cast<size_t>(frame_rate_local));

      if (ImGui::ImageButton(_play_icon)) {
	_play_video.store(true);
      }
      ImGui::SameLine();
      if (ImGui::ImageButton(_pause_icon)) {
	_play_video.store(false);
      }
      ImGui::SameLine();
      if (ImGui::ImageButton(_prev_icon)) {
	auto frame_index_local = _frame_index.load();
	if (frame_index_local == 0)
	  frame_index_local = _envelopes.size() - 1;
	else
	  frame_index_local -= 1;
	_frame_index.store(frame_index_local);
      }
      ImGui::SameLine();
      if (ImGui::ImageButton(_next_icon)) {
	auto frame_index_local = _frame_index.load();
	frame_index_local = (frame_index_local + 1) % _envelopes.size();
	_frame_index.store(frame_index_local);
      }
      ImGui::End();
    }
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
    int env_cols  = _envelopes[0].cols;
    int env_rows  = _envelopes[0].rows;

    auto output_gray  = cv::Mat(env_rows, env_cols, CV_32FC1);
    auto output_quant = cv::Mat(env_rows, env_cols, CV_8UC1);
    auto output_rgba  = cv::Mat(env_rows, env_cols, CV_8UC4);
    auto log_image    = cv::Mat(env_rows, env_cols, CV_32FC1);

    _image_processing_lock.lock();
    _image_processing.apply(log_image, _mask, output_gray, param);
    _image_processing_lock.unlock();

    this->dynamic_range_adjustment(output_gray, _mask,
				   _dynamic_range.load(),
				   _reject.load());
    this->quantize(output_gray, output_quant);

    cv::cvtColor(output_quant, output_rgba, cv::COLOR_GRAY2RGBA);
    _preview_buffer.update(output_rgba.ptr());
  }

  void
  VideoPlayer::
  export_files(std::string const& export_path,
	       blaze::DynamicVector<double> const& best_param)
  {
    auto parameter_local = blaze::DynamicVector<double>();
    size_t env_rows      = _envelopes[0].rows;
    size_t env_cols      = _envelopes[0].cols;
    auto output_image    = cv::Mat(env_rows, env_cols, CV_32F);

    auto output_sequence = std::vector<cv::Mat>(_envelopes.size());
    for (size_t i = 0; i < _envelopes.size(); ++i)
    {
      _image_processing_lock.lock();
      _image_processing.apply(_envelopes[i], _mask, output_image, best_param);
      _image_processing_lock.unlock();
      cv::imwrite(export_path + "/processed_" + std::to_string(i) + ".pfm", output_image);
    }
  }

  void
  VideoPlayer::
  toggle_preview()
  {
    _show_preview = !_show_preview;
  }
}
