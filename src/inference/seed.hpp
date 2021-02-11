
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

#ifndef __US_GALLERY_SEED_HPP__
#define __US_GALLERY_SEED_HPP__

#include <algorithm>
#include <array>
#include <functional>
#include <random>

inline std::mt19937_64
generate_seed(uint64_t seed)
{
  auto seeder = std::mt19937_64(seed);
  std::array<int,624> seed_data;
  std::random_device r;
  std::generate_n(seed_data.data(), seed_data.size(), std::ref(seeder));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  auto rng  = std::mt19937_64();
  rng.seed(seq);
  return rng;
}

#endif
