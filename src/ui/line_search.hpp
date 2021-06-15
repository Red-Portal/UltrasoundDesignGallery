
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


#ifndef __US_GALLERY_LINESEARCH_HPP__
#define __US_GALLERY_LINESEARCH_HPP__

#include <SFML/Graphics.hpp>	
#include "../math/blaze.hpp"
//#include "optimization_manager.hpp"

namespace usdg
{
  class LineSearch
  {
  private:
    float _slider_pos;
    float _slider_fine_step;

    sf::Image   _select_icon_image;
    sf::Image   _select_icon_disabled_image;
    sf::Texture _select_icon;
    sf::Texture _slider_next_icon;
    sf::Texture _slider_prev_icon;

    bool   _select_button_pressed;

    ImVec4 _select_button_disabled_color;      
    ImVec4 _select_button_enabled_hovered_color;      
    ImVec4 _select_button_enabled_active_color;      

    //usdg::OptimizationManager _opt_manager;

    void color_button_disabled();

    void render_select_button();

    void render_slider_fine_control();

  public:
    LineSearch();

    void render();

    double selected_parameter() noexcept;

    bool is_select_pressed() noexcept;

    void enable_select_button() noexcept;
  };
}

#endif
