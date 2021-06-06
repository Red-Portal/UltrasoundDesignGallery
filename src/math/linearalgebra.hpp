
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
#include "lu.hpp"
#include "blaze.hpp"

#include <optional>
#include <iostream>

namespace usdg
{
  template <typename CholType>
  inline blaze::DynamicVector<double> 
  solve(usdg::Cholesky<CholType> Achol,
	blaze::DynamicVector<double> const& b)
  {
    auto Ux = blaze::evaluate(blaze::solve(Achol.L, b));
    auto U  = blaze::declupp(blaze::trans(Achol.L));
    auto x  = blaze::solve(U, Ux);
    return x;
  }

  inline blaze::DynamicVector<double> 
  solve(usdg::LU const& Alu,
	blaze::DynamicVector<double> const& b)
  {
    auto Pb = blaze::evaluate(Alu.Pt * b);
    auto Ux = blaze::evaluate(blaze::solve(Alu.L, Pb));
    auto x  = blaze::solve(Alu.U, Ux);
    return x;
  }

  inline double
  invquad(usdg::Cholesky<usdg::DenseChol> const& chol,
	  blaze::DynamicVector<double> const& x)
  {
    auto y = blaze::solve(chol.L, x);
    return blaze::dot(y, y);
  }

  inline blaze::DynamicVector<double>
  invquad_batch(usdg::Cholesky<usdg::DenseChol> const& chol,
		blaze::DynamicMatrix<double> const& X)
  {
    auto Y   = blaze::evaluate(blaze::solve(chol.L, X));
    auto YpY = blaze::evaluate(Y%Y);
    return blaze::trans(blaze::sum<blaze::columnwise>(YpY));
  }

  inline double
  invquad(usdg::Cholesky<usdg::DiagonalChol> const& chol,
	  blaze::DynamicVector<double> const& x)
  {
    return blaze::dot(x / chol.A, x);
  }

  inline blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>
  inverse(usdg::Cholesky<usdg::DenseChol> const& chol)
  {
    auto Linv = chol.L;
    blaze::invert(Linv);
    return blaze::trans(Linv) * Linv;
  }

  // inline blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>
  // inverse(usvg::Cholesky<usvg::DiagonalChol> const& chol)
  // {
  //   return 1.0 / chol.A;
  // }

  // inline double
  // invquad(usvg::LU const& IpWK,
  // 	  blaze::SymmetricMatrix<blaze::DynamicMatrix<double>> const& K,
  // 	  blaze::DynamicMatrix<double> const& WK,
  // 	  blaze::DynamicVector<double> const& x)
  // /*
  //  * Note: ( K^{-1} + W )^{-1} = K ( I - ( I + W K )^{-1} W K ) 
  //  *                           = K ( I - B^{-1} W K ) 
  //  */
  // {
  //   auto WKx       = WK * x;
  //   auto BinvWKx   = usvg::solve(IpWK, WKx);
  //   auto KinvWinvx = K * (x - BinvWKx);
  //   return blaze::dot(x, KinvWinvx);
  // }

  inline double
  logtrace(blaze::DynamicMatrix<double> const& A)
  /*
   * Compute log of matrix trace using Kahan's method.
   */
  {
    size_t n_dims  = A.rows();
    double logdiag = 0.0;
    double c       = 0.0;
    for (size_t i  = 0; i < n_dims; ++i)
    {
      double y = log(A(i, i)) - c;
      double t = logdiag + y;
      c        = (t - logdiag) - y;
      logdiag  = t;
    }
    return logdiag;
  }

  inline double
  logtrace(blaze::DynamicVector<double> const& A)
  /*
   * Compute log of diagonal matrix trace using Kahan's method.
   */
  {
    size_t n_dims  = A.size();
    double logdiag = 0.0;
    double c       = 0.0;
    for (size_t i  = 0; i < n_dims; ++i)
    {
      double y = log(A[i]) - c;
      double t = logdiag + y;
      c        = (t - logdiag) - y;
      logdiag  = t;
    }
    return logdiag;
  }

  inline double
  logabstrace(blaze::DynamicMatrix<double> const& A)
  /*
   * Compute log absolute of trace using LU decomposition using Kahan's method.
   */
  {
    size_t n_dims  = A.rows();
    double logdiag = 0.0;
    double c       = 0.0;
    for (size_t i  = 0; i < n_dims; ++i)
    {
      double y = log(abs(A(i,i))) - c;
      double t = logdiag + y;
      c        = (t - logdiag) - y;
      logdiag  = t;
    }
    return logdiag;
  }

  template<typename CholType>
  inline double
  logdet(usdg::Cholesky<CholType> const& chol)
  /*
   * Compute log determinant using Cholesky decomposition
   * Summation uses Kahan's method.
   */
  {
    auto L_diag = usdg::logtrace(chol.L);
    return 2*L_diag;
  }

  inline double
  logdet(usdg::LU const& lu)
  /*
   * Compute log determinant using Cholesky decomposition
   * Summation uses Kahan's method.
   */
  {
    auto L_diag = usdg::logtrace(lu.L);
    auto U_diag = usdg::logtrace(lu.U);
    return L_diag + U_diag;
  }

  inline double
  logabsdet(usdg::LU const& lu)
  /*
   * Compute log determinant using Cholesky decomposition
   * Summation uses Kahan's method.
   */
  {
    auto L_diag = usdg::logabstrace(lu.L);
    auto U_diag = usdg::logabstrace(lu.U);
    return L_diag + U_diag;
  }
}

#endif
