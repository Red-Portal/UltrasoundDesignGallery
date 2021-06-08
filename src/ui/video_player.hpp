
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

#include <SFML/Graphics.hpp>	

namespace usdg
{
  class VideoPlayer
  {
    sf::Image _image;
    sf::Texture _front_buffer;

    sf::Texture _play_icon;
    sf::Texture _pause_icon;
    sf::Texture _stop_icon;
    sf::Texture _loop_icon;
    
  public:
    VideoPlayer(std::string const& fpath) noexcept;

    void render();
  };
}

#endif