
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

#include <imgui.h>
#include <imgui-SFML.h>
#include <string>
#include <iostream>

#include "video_player.hpp"

#define ICON(NAME) USGALLERY_ROOT "/resource/icons/" NAME

namespace usdg
{
  VideoPlayer::VideoPlayer(std::string const& fpath) noexcept
    : _image(),
      _front_buffer(),
      _play_icon(),
      _pause_icon(),
      _stop_icon(),
      _loop_icon()
  {
    _image.loadFromFile(fpath);
    _front_buffer.loadFromImage(_image);

    auto image_size  = _image.getSize();
    auto desktopMode = sf::VideoMode::getDesktopMode();
    auto width       = std::min(desktopMode.width,  image_size.x);
    auto height      = std::min(desktopMode.height, image_size.y);
    auto window_size = ImVec2(static_cast<float>(width), static_cast<float>(height));
    ImGui::Begin("Video");
    ImGui::SetWindowSize(window_size);
    ImGui::End();
    
    _play_icon.loadFromFile(ICON("play.png"));
    _pause_icon.loadFromFile(ICON("pause.png"));
    _stop_icon.loadFromFile(ICON("stop.png"));
    _loop_icon.loadFromFile(ICON("loop.png"));
  }

  void
  VideoPlayer::render()
  {
    if(ImGui::Begin("Video"))
    {
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
}
