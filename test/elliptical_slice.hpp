

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

#ifndef __US_GALLERY_ELLIPTICAL_HPP__
#define __US_GALLERY_ELLIPTICAL_HPP__

#include "../src/inference/ess.hpp"
#include "../src/misc/blaze.hpp"
#include "../src/misc/mvnormal.hpp"

template <typename Rng, typename LoglikeFunc, typename CholType>
inline blaze::DynamicMatrix<double>
elliptical_slice(Rng& prng,
		 size_t n_samples,
		 size_t n_burnin,
		 blaze::DynamicVector<double> const& x0,
		 LoglikeFunc loglike,
		 usdg::MvNormal<CholType> const& prior_dist)
{
  size_t n_dims = x0.size();
  auto x        = x0;
  auto p        = loglike(x0);
  auto samples  = blaze::DynamicMatrix<double>(n_dims, n_samples);

  for (size_t i = 0; i < n_burnin; ++i)
  { /* burnin */
    auto [x_prop, p_prop, n_props] = usdg::ess_transition(
      prng, loglike, x, p, prior_dist);
    x = x_prop;
    p = p_prop;
  }

  size_t n_total_props = 0;
  for (size_t i = 0; i < n_samples; ++i)
  {
    auto [x_prop, p_prop, n_props] = usdg::ess_transition(
      prng, loglike, x, p, prior_dist);
    x = x_prop;
    p = p_prop;
    n_total_props += n_props;
    blaze::column(samples, i) = x;
  }
  return samples;
}

#endif
