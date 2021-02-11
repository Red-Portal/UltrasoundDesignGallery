
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
  template <typename DMat, typename DVec>
  inline double
  invquad(blaze::LowerMatrix<DMat> const& A_chol, DVec const& x)
  {
    auto y = blaze::solve(A_chol, x);
    return blaze::dot(y, y);
  }

  template <typename DMat>
  inline double
  logdet(blaze::LowerMatrix<DMat> const& A_chol)
  /*
   * Compute log determinant using Cholesky decomposition
   * Summation uses Kahan's method.
   */
  {
    size_t n_dims = A_chol.rows();
    double L_diag = 0.0;
    double c      = 0.0;
    for (size_t i = 0; i < n_dims; ++i)
    {
      double y = log(A_chol(i, i)) - c;
      double t = L_diag + y;
      c        = (t - L_diag) - y;
      L_diag   = t;
    }
    return 2 * L_diag;
  }

  inline bool
  potrf_nothrow(blaze::DynamicMatrix<double>& A, char uplo)
  {
    if( !blaze::isSquare( A ) )
    {
      return false;
    }

    if( uplo != 'L' && uplo != 'U' )
    {
      return false;
    }

    blaze::blas_int_t n   ( blaze::numeric_cast<blaze::blas_int_t>( A.rows()    ) );
    blaze::blas_int_t lda ( blaze::numeric_cast<blaze::blas_int_t>( A.spacing() ) );
    blaze::blas_int_t info( 0 );

    if( n == 0 )
    {
      return true;
    }

    if( blaze::IsRowMajorMatrix_v<blaze::DynamicMatrix<double>> )
    {
      ( uplo == 'L' )?( uplo = 'U' ):( uplo = 'L' );
    }

    blaze::potrf( uplo, n, A.data(), lda, &info );

    BLAZE_INTERNAL_ASSERT( info >= 0, "Invalid argument for Cholesky decomposition" );

    if( info > 0 )
    {
      return false;
    }
    else
    {
      return true;
    }
  }

  inline std::optional<usvg::Cholesky>
  cholesky_nothrow(blaze::DynamicMatrix<double> const& A)
  {
    size_t n = A.rows();
    auto L_buf   = blaze::DynamicMatrix<double>(n, n); 

    if( blaze::IsRowMajorMatrix_v<blaze::DynamicMatrix<double>> )
    {
      for( size_t i = 0UL; i < n; ++i ) {
	for( size_t j = 0UL; j <= i ; ++j ) {
	  L_buf(i,j) = A(i,j);
	}
      }
    }
    else
    {
      for( size_t j = 0UL; j < n; ++j ) {
	for( size_t i = j; i < n; ++i ) {
	  L_buf(i,j) = A(i,j);
	}
      }
    }

    bool success = potrf_nothrow(L_buf, 'L');
    if(success)
    {
      auto L = blaze::LowerMatrix<decltype(L_buf)>(
	blaze::decllow(L_buf));
      return Cholesky{A, L};
    }
    else
    {
      return std::nullopt;
    }
  }
}

#endif
