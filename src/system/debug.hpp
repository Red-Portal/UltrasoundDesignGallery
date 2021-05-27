
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

#ifndef __US_GALLERY_DEBUG_HPP__
#define __US_GALLERY_DEBUG_HPP__

#include "../math/blaze.hpp"

#include <matplotlib-cpp/matplotlibcpp.h>
#include <spdlog/spdlog.h>
//#include <spdlog/sinks/stdout_color_sinks.h>

// auto console  = spdlog::stdout_color_mt("console");
// spdlog::set_level(spdlog::level::info);
// auto logger  = spdlog::get("console");


namespace usdg
{
  constexpr const char*
  file_name(const char* path) {
    /* 
     * Compile time path stripping of the __FILE__ macro.
     * Written by pexeer,
     * retrieved from https://stackoverflow.com/questions/31050113
     */
    const char* file = path;
    while (*path) {
      if (*path++ == '/') {
	file = path;
      }
    }
    return file;
  }

  template <typename Func>
  inline void
  render_function(Func func)
  {
    std::vector<std::vector<double>> x_plt, y_plt, z_plt;
    for (double i = 0; i <= 1.0;  i += 0.03) {
      std::vector<double> x_row, y_row, z_row;
      for (double j = 0; j <= 1.0; j += 0.03) {
	x_row.push_back(i);
	y_row.push_back(j);
	z_row.push_back(
	  func(blaze::DynamicVector<double>({i, j})));
      }
      x_plt.push_back(x_row);
      y_plt.push_back(y_row);
      z_plt.push_back(z_row);
    }
    matplotlibcpp::plot_surface(x_plt,y_plt,z_plt);
    matplotlibcpp::show();
  }
}

#endif
