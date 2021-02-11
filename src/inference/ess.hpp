
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

#ifndef __US_GALLERY_ESS_HPP__
#define __US_GALLERY_ESS_HPP__

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>

#include <algorithm>
#include <random>
#include <stats.hpp>

namespace infer
{
  template <typename Rng, typename Loglike, typename Prior>
  inline blaze::DynamicVector<double>
  ess_transition(Rng rng,
		 Loglike p,
		 blaze::DynamicVector<double> const& prev_x,
		 double prev_like,
		 Prior  prior,
		 blaze::DynamicVector<double> const& prior_mean,
		 blaze::DynamicVector<double> const& prior_cov_chol,
		 size_t n_samples,
		 size_t n_burn)
  {
    size_t n_dims = prior_mean.size();
    auto z  = stats::rnorm(n_dims, 1u);
    auto nu = prior_cov_chol;
    //auto samples = blaze::DynamicVector<double>(n_samples / n_thin);
  }
}

#endif
