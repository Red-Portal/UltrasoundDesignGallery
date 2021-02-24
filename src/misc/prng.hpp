
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

#ifndef __US_GALLERY_PRNG_HPP__
#define __US_GALLERY_PRNG_HPP__

#include <istream>

#include <boost/random/philox.hpp>
#include <boost/random/counter_based_engine.hpp>

namespace usdg
{
  using Philox        = boost::random::philox<4, uint64_t>;
  using CounterEngine = boost::random::counter_based_engine<uint64_t, Philox, 32>; 

  using Random123 = CounterEngine;
}

#endif
