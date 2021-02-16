
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

#ifndef __ULTRASOUND_DESIGN_GALLERY_UTILS_HPP__
#define __ULTRASOUND_DESIGN_GALLERY_UTILS_HPP__

#include "../src/gp/gp_prior.hpp"
#include "../src/gp/kernel.hpp"
#include "../src/misc/cholesky.hpp"
#include "../src/misc/mvnormal.hpp"

template <typename Rng>
inline blaze::DynamicMatrix<double>
generate_mvsamples(Rng& rng, size_t n_dims, size_t n_points)
{
  auto data = blaze::DynamicMatrix<double>(n_dims, n_points);
  for (size_t i = 0; i < n_points; ++i)
  {
    blaze::column(data, i) = usvg::rmvnormal(rng, n_dims);
  }
  return data;
}

template <typename Rng>
inline blaze::DynamicVector<double>
sample_gp_prior(Rng& prng,
		usvg::Matern52 const& kernel,
		blaze::DynamicMatrix<double> const& points)
{
  auto K      = usvg::compute_gram_matrix(kernel, points);
  auto K_chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( K_chol = usvg::cholesky_nothrow(K).value() );
  
  auto Z = usvg::rmvnormal(prng, K.rows());
  auto y = K_chol.L * Z;
  return y;
}

#endif
