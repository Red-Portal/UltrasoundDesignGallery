
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
#include <iostream>

#include "line_search.hpp"
#include "utils.hpp"

namespace usdg
{
  void
  LineSearch::
  color_button_disabled()
  {
    auto style  = &ImGui::GetStyle();
    auto colors = style->Colors;
    auto color  = colors[ImGuiCol_Button];
    color.w     = 30u;
    _select_button_disabled_color = color;
  }

  LineSearch::
  LineSearch()
    : _slider_pos(0.5f),
      _slider_fine_step(0.01f),
      _select_icon_image(),
      _select_icon_disabled_image(),
      _select_icon(),
      _slider_next_icon(),
      _slider_prev_icon(),
      _select_button_pressed(false),
      _select_button_disabled_color()
  {
    auto desktopMode = sf::VideoMode::getDesktopMode();
    float width      = std::min(static_cast<float>(desktopMode.width)*0.8f, 450.0f);
    float height     = 200;
    auto window_size = ImVec2(width, height);
    ImGui::Begin("Line Search");
    ImGui::SetWindowSize(window_size);
    ImGui::End();

    _select_icon_image.loadFromFile(ICON("check_white.png"));
    _select_icon.loadFromImage(_select_icon_image);
    _select_icon_disabled_image.loadFromFile(ICON("check_white.png"));
    auto icon_size = _select_icon_disabled_image.getSize();
    for (unsigned int x = 0; x < icon_size.x; ++x)
    {
      for (unsigned int y = 0; y < icon_size.y; ++y)
      {
	auto pixel = _select_icon_disabled_image.getPixel(x, y);
	pixel.a    = 30u;
	_select_icon_disabled_image.setPixel(x, y, pixel);
      }
    }

    _slider_next_icon.loadFromFile(ICON("next_white.png"));
    _slider_prev_icon.loadFromFile(ICON("prev_white.png"));
  }

  void
  LineSearch::
  render_select_button()
  {
    ImGui::Text("approve setting");
    ImGui::SameLine();
    if (!_select_button_pressed)
    {
      if (ImGui::ImageButton(_select_icon))
      {
	_select_button_pressed = true;
	_select_icon.loadFromImage(_select_icon_disabled_image);
      }
    }
    else
    {
      ImGui::PushID("select");
      ImGui::PushStyleColor(ImGuiCol_Button,        _select_button_disabled_color);
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, _select_button_disabled_color);
      ImGui::PushStyleColor(ImGuiCol_ButtonActive,  _select_button_disabled_color);
      ImGui::ImageButton(_select_icon);
      ImGui::PopStyleColor(3);
      ImGui::PopID();
      // if (ImGui::IsItemClicked(0))
      // {
      // 	_select_button_pressed = !_select_button_pressed;
      // 	_select_icon.loadFromImage(_select_icon_image);
      // }
    }
  }

  void
  LineSearch::render()
  {
    if(ImGui::Begin("Line Search"))
    {
      ImGui::SliderFloat("Settings", &_slider_pos, 0.0, 1.0);

      if (ImGui::TreeNode("Slider Fine Control"))
      {
	ImGui::PushItemWidth(52);
	ImGui::InputFloat("step size", &_slider_fine_step);
	ImGui::PopItemWidth();
	ImGui::SameLine();
	if (ImGui::ImageButton(_slider_prev_icon)) {
	  _slider_pos -= _slider_fine_step;
	}
	ImGui::SameLine();
	if (ImGui::ImageButton(_slider_next_icon)) {
	  _slider_pos += _slider_fine_step;
	}
	ImGui::TreePop();
      }

      //ImGui::Separator();
      this->render_select_button();
    }
    ImGui::End();
  }

  blaze::DynamicVector<double>
  LineSearch::selected_parameter()
  {
    //return _opt_manager->query(_slider_pos);
    return {};
  }
}

