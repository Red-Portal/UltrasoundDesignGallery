
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

#ifndef __US_GALLERY_LU_HPP__
#define __US_GALLERY_LU_HPP__

#include <blaze/math/LowerMatrix.h>
#include <blaze/math/DynamicMatrix.h>

namespace usvg
{
  struct LU
  {
    /* PLU for column major (refer to the blaze docs.) */
    blaze::DynamicMatrix<double> A;
    blaze::DynamicMatrix<double> _L;
    blaze::DynamicMatrix<double> _U;
    blaze::LowerMatrix<blaze::DynamicMatrix<double>> L;
    blaze::UpperMatrix<blaze::DynamicMatrix<double>> U;
    blaze::DynamicMatrix<double> Pt;

    inline
    LU();

    template <typename DenseMat,
	      typename UMat,
	      typename LMat,
	      typename PMat>
    inline
    LU(DenseMat&& A_, LMat&& L_, UMat&& U_, PMat&& P_);
  };

  inline
  LU::
  LU()
    : A(), _L(), _U(), L(), U(), Pt()
  { }

    template <typename DenseMat,
	      typename UMat,
	      typename LMat,
	      typename PMat>
  inline
  LU::
  LU(DenseMat&& A_, LMat&& L_, UMat&& U_, PMat&& Pt_)
    : A(std::forward<DenseMat>(A_)),
      _L(std::forward<DenseMat>(L_)),
      _U(std::forward<DenseMat>(U_)),
      L(blaze::decllow(_L)),
      U(blaze::declupp(_U)),
      Pt(Pt_)
  { }

  inline LU
  lu(blaze::DynamicMatrix<double> const& A)
  {
    size_t n   = A.rows();
    auto L_buf = blaze::DynamicMatrix<double>(n, n, 0.0); 
    auto U_buf = blaze::DynamicMatrix<double>(n, n, 0.0); 
    auto P     = blaze::DynamicMatrix<double>(n, n, 0.0); 

    lu(A, L_buf, U_buf, P);

    return LU(A, L_buf, U_buf, blaze::trans(P));
  }
}

#endif
