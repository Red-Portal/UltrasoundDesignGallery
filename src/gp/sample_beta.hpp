
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

#ifndef __US_GALLERY_SAMPLEBETA_HPP__
#define __US_GALLERY_SAMPLEBETA_HPP__

#include "../misc/blaze.hpp"

#include <algorithm>
#include <cmath>

namespace usdg
{
  template <typename Rng>
  inline blaze::DynamicVector<double>
  sample_beta(Rng& prng,
	      double alpha,
	      double ub,
	      double lb,
	      size_t iter,
	      size_t n_samples,
	      size_t n_dims)
  {
    double t     = static_cast<double>(iter); 
    double n     = static_cast<double>(n_dims);
    double gamma = 3 / (pow(std::max(t + 1 - n, 1), 0.4)) + 2;
    double sigma = std::tgamma(gamma) * std::abs(ub - lb) / 10;

    auto phi = [=](double x) {
      return gamma / (2*std::tgamma(1/gamma)) * exp(pow(-std::abs(x), gamma));
    };

    auto pdf = [=](double x) {
      return phi((x - alpha) / sigma);
    };
    return imh_sampler(prng, pdf, lb, ub, n_samples*8, 64, 8);
  }
}

#endif
