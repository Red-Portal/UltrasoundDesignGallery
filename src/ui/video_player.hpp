
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

#ifndef __US_GALLERY_VIDEOPLAYER_HPP__
#define __US_GALLERY_VIDEOPLAYER_HPP__

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <SFML/Graphics.hpp>	
#include <opencv4/opencv2/core/utility.hpp>

#include "../math/blaze.hpp"
#include "../custom_image_processing.hpp"

namespace usdg
{
  class VideoPlayer
  { /* Danger: This class is very dirty and has a few lock-related caviats. */
  private:
    cv::Mat                      _image_base;
    cv::Mat                      _render_buffer;
    cv::Mat                      _back_buffer;
    sf::Texture                  _front_buffer;
    std::mutex                   _buffer_lock;

    blaze::DynamicVector<double> _parameter;
    std::mutex                   _parameter_lock;
    std::thread                  _imageproc_thread;
    std::atomic<bool>            _terminate_thread;

    usdg::CustomImageProcessing  _image_processing;
    std::mutex                   _image_processing_lock;

    sf::Texture                  _preview_buffer;
    bool                         _show_preview;

    sf::Texture _play_icon;
    sf::Texture _pause_icon;
    sf::Texture _stop_icon;
    sf::Texture _loop_icon;

    void quantize(cv::Mat const& src, cv::Mat& dst);

  public:
    VideoPlayer(blaze::DynamicVector<double> const& param_init,
		std::string const& fpath);

    ~VideoPlayer();

    void render();

    void update_parameter(blaze::DynamicVector<double> const& param);

    void update_preview(blaze::DynamicVector<double> const& param);

    void toggle_preview();
  };
}

#endif
