
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
#include <algorithm>

#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/Window/Keyboard.hpp>

#include "line_search.hpp"
#include "utils.hpp"
#include "spinner.hpp"

namespace usdg
{
  LineSearch::
  LineSearch()
    : _slider_pos(0.5f),
      _slider_fine_step(0.01f),
      _macro_positions({0.0f, 0.33f, 0.66f, 1.0f}),
      _select_icon_image(),
      _select_icon_disabled_image(),
      _select_icon(),
      _slider_next_icon(),
      _slider_prev_icon(),
      _iteration(0),
      _select_button_pressed(false)
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
    for (unsigned int x = 0; x < icon_size.x; ++x) {
      for (unsigned int y = 0; y < icon_size.y; ++y) {
	auto pixel = _select_icon_disabled_image.getPixel(x, y);
	pixel.a    = 100u;
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
    if(_select_button_pressed)
    {
      auto col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
      ImGui::Spinner("##spinner", 10, 6.0, col);
    }
    else
    {
      auto& io = ImGui::GetIO();
      if(ImGui::ImageButton(_select_icon)
	 || (ImGui::IsItemFocused()
	     && ImGui::IsKeyPressed(io.KeyMap[ImGuiKey_Enter])))
      {
	_select_icon.loadFromImage(_select_icon_disabled_image);
	_select_button_pressed = true;
      }
    }
    ImGui::SameLine();
    ImGui::Text("approve setting");
  }


  void
  LineSearch::
  render_comparison_macro()
  {
    bool input_is_active = false;
    if (ImGui::TreeNode("Comparison Macro"))
    {
      ImGui::PushItemWidth(60);
      ImGui::InputFloat("setting 1", &_macro_positions[0]);
      input_is_active = input_is_active || ImGui::IsItemActive();
      _macro_positions[0] = std::clamp(_macro_positions[0], 0.0f, 1.0f);
      ImGui::SameLine();
      if (ImGui::SmallButton("write##1"))
      {
	_macro_positions[0] = _slider_pos;
      }

      ImGui::InputFloat("setting 2", &_macro_positions[1]);
      input_is_active = input_is_active || ImGui::IsItemActive();
      _macro_positions[1] = std::clamp(_macro_positions[1], 0.0f, 1.0f);
      ImGui::SameLine();
      if (ImGui::SmallButton("write##2"))
      {
	_macro_positions[1] = _slider_pos;
      }

      ImGui::InputFloat("setting 3", &_macro_positions[2]);
      input_is_active = input_is_active || ImGui::IsItemActive();
      _macro_positions[2] = std::clamp(_macro_positions[2], 0.0f, 1.0f);
      ImGui::SameLine();
      if (ImGui::SmallButton("write##3"))
      {
	_macro_positions[2] = _slider_pos;
      }

      ImGui::InputFloat("setting 4", &_macro_positions[3]);
      input_is_active = input_is_active || ImGui::IsItemActive();
      _macro_positions[3] = std::clamp(_macro_positions[3], 0.0f, 1.0f);
      ImGui::SameLine();
      if (ImGui::SmallButton("write##4"))
      {
	_macro_positions[3] = _slider_pos;
      }

      if (!input_is_active)
      {
	if(ImGui::IsKeyPressed(sf::Keyboard::Num1))
	  _slider_pos = _macro_positions[0];
	else if(ImGui::IsKeyPressed(sf::Keyboard::Num2))
	  _slider_pos = _macro_positions[1];
	else if(ImGui::IsKeyPressed(sf::Keyboard::Num3))
	  _slider_pos = _macro_positions[2];
	else if(ImGui::IsKeyPressed(sf::Keyboard::Num4))
	  _slider_pos = _macro_positions[3];
      }

      ImGui::PopItemWidth();
      ImGui::TreePop();
    }
  }

  void
  LineSearch::
  render_slider_fine_control()
  {
    auto& io = ImGui::GetIO();
    if (ImGui::TreeNode("Slider Fine Control"))
    {
      ImGui::PushItemWidth(52);
      ImGui::InputFloat("step size", &_slider_fine_step);
      ImGui::PopItemWidth();
      if (ImGui::ImageButton(_slider_prev_icon)
	  || (ImGui::IsItemFocused()
	      && ImGui::GetKeyPressedAmount(io.KeyMap[ImGuiKey_Enter], 0, 0.1f)) > 0)
      {
	_slider_pos = std::clamp(_slider_pos - _slider_fine_step, 0.0f, 1.0f);
      }
      ImGui::SameLine();
      if(ImGui::ImageButton(_slider_next_icon)
	 || (ImGui::IsItemFocused()
	     && ImGui::GetKeyPressedAmount(io.KeyMap[ImGuiKey_Enter], 0, 0.1f)) > 0)
      {
	_slider_pos = std::clamp(_slider_pos + _slider_fine_step, 0.0f, 1.0f);
      }
      ImGui::SameLine();
      ImGui::Text("single step");
      ImGui::TreePop();
    }
  }

  void
  LineSearch::
  render_slider()
  {
    ImGui::SliderFloat("setting", &_slider_pos, 0.0, 1.0);
    if(ImGui::IsItemFocused())
    {
      auto& io = ImGui::GetIO();
      if (ImGui::GetKeyPressedAmount(io.KeyMap[ImGuiKey_LeftArrow], 0, 0.1f) > 0)
      {
	_slider_pos = std::clamp(_slider_pos - _slider_fine_step, 0.0f, 1.0f);
      }
      else if (ImGui::GetKeyPressedAmount(io.KeyMap[ImGuiKey_RightArrow], 0, 0.1f) > 0)
      {
	_slider_pos = std::clamp(_slider_pos + _slider_fine_step, 0.0f, 1.0f);
      }
    }
  }

  void
  LineSearch::
  render()
  {
    if(ImGui::Begin("Line Search"))
    {
      this->render_slider();
      this->render_slider_fine_control();
      this->render_comparison_macro();
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();

      ImGui::PushItemWidth(38);
      ImGui::LabelText("iteration##linesearch", "%zu", _iteration);
      this->render_select_button();
    }
    ImGui::End();
  }

  double
  LineSearch::
  selected_parameter() noexcept
  {
    return _slider_pos;
  }

  bool
  LineSearch::
  is_select_pressed() noexcept
  {
    return _select_button_pressed;
  }

  void
  LineSearch::
  enable_select_button() noexcept
  {
    _select_button_pressed = false;
    _select_icon.loadFromImage(_select_icon_image);
  }

  void
  LineSearch::
  disable_select_button() noexcept
  {
    _select_button_pressed = true;
  }

  void
  LineSearch::
  update_iteration(size_t iteration) noexcept
  {
    _iteration = iteration;
    _slider_pos = 0.5;
  }
}

