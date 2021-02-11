
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

#ifndef __US_GALLERY_LINEARALGEBRA_HPP__
#define __US_GALLERY_LINEARALGEBRA_HPP__

#include "cholesky.hpp"

#include <blaze/math/LowerMatrix.h>
#include <blaze/math/DenseMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DenseVector.h>

#include <optional>

namespace usvg
{
  inline double
  invquad(usvg::Cholesky<usvg::DenseChol> const& chol,
	  blaze::DynamicVector<double> const& x)
  {
    auto y = blaze::solve(chol.L, x);
    return blaze::dot(y, y);
  }

  inline double
  invquad(usvg::Cholesky<usvg::DiagonalChol> const& chol,
	  blaze::DynamicVector<double> const& x)
  {
    return blaze::dot(x / chol.A, x);
  }

  inline double
  logdet(usvg::Cholesky<usvg::DenseChol> const& chol)
  /*
   * Compute log determinant using Cholesky decomposition
   * Summation uses Kahan's method.
   */
  {
    size_t n_dims = chol.L.rows();
    double L_diag = 0.0;
    double c      = 0.0;
    for (size_t i = 0; i < n_dims; ++i)
    {
      double y = log(chol.L(i, i)) - c;
      double t = L_diag + y;
      c        = (t - L_diag) - y;
      L_diag   = t;
    }
    return 2 * L_diag;
  }

  inline double
  logdet(usvg::Cholesky<usvg::DiagonalChol> const& chol)
  /*
   * Compute log determinant using Cholesky decomposition
   * Summation uses Kahan's method.
   */
  {
    size_t n_dims = chol.L.size();
    double L_diag = 0.0;
    double c      = 0.0;
    for (size_t i = 0; i < n_dims; ++i)
    {
      double y = log(chol.L[i]) - c;
      double t = L_diag + y;
      c        = (t - L_diag) - y;
      L_diag   = t;
    }
    return 2 * L_diag;
  }
}

#endif
