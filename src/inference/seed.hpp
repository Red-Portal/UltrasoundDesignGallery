
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
