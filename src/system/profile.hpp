
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

#ifndef __US_GALLERY_PROFILE_HPP__
#define __US_GALLERY_PROFILE_HPP__

#include <chrono>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>

namespace usdg
{
  using microsecond = std::chrono::duration<std::micro, double>;
  using millisecond = std::chrono::duration<std::milli, double>;
  using second      = std::chrono::duration<std::ratio<1,1>, double>;
  using clock       = std::chrono::steady_clock;
  using namespace std::string_literals;

  class Profiler
  {
    std::unordered_map<std::string, std::chrono::time_point<clock>> _start;
    std::unordered_map<std::string, usdg::second> _duration;

  public:
    inline void
    start(std::string_view region_id);

    inline void
    stop(std::string_view region_id);

    friend std::ostream&
    operator<<(std::ostream& os, Profiler const& profiler);
  };

  inline void
  Profiler::
  start(std::string_view region_id)
  {
    _start[region_id] = usdg::clock::now();
  }

  inline void
  Profiler::
  stop(std::string_view region_id)
  {
    auto stop     = usdg::clock::now();
    auto duration = std::chrono::duration_cast<usdg::second>(
      stop - this->_start[region_id]);

    if(duration.count() < 0)
    {
      throw std::runtime_error("Negative execution time measured");
    }
    this->_duration[region_id] = duration;
  }

  inline std::ostream&
  operator<<(std::ostream& os, Profiler const& profiler)
  {
    for (auto const& [key, val] : profiler._duration)
    {
      std::cout << key << ": " << val.count << "sec\n";
    }
  }
}

#endif
