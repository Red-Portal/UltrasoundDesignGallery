
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

#include <catch2/catch_all.hpp>
#define BLAZE_USE_DEBUG_MODE 1

#include "../src/math/root.hpp"

#include <cmath>

TEST_CASE("find root with Brent's method", "[root]")
{
  auto f = [](double x){
    return (x + 3)*pow(x - 1, 2);
  };
  auto x = usdg::find_zero( -10, 10, 1e-10, f );
  REQUIRE( abs(f(x)) < 1e-5 );
}
