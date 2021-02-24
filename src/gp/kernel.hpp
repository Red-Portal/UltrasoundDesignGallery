
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

#ifndef __US_GALLERY_KERNEL_HPP__
#define __US_GALLERY_KERNEL_HPP__

#include <blaze/math/DynamicVector.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/Subvector.h>

#include <cmath>

namespace usdg
{
  struct Matern52
  {
    double sigma;
    blaze::DynamicVector<double> ardscales;

    inline Matern52();

    template <typename VecType>
    inline Matern52(VecType const& theta);
    
    template <typename VecType>
    inline Matern52(double sigma, VecType const& linescales);

    template<typename VecLHSType,
	     typename VecRHSType>
    inline double operator()(VecLHSType const& x,
			     VecRHSType const& y) const noexcept;

    inline blaze::DynamicVector<double> vector() const;
  };

  inline
  Matern52::
  Matern52()
    : sigma(), ardscales()
  {}

  template <typename VecType>
  inline
  Matern52::
  Matern52(VecType const& theta)
    : sigma(theta[0]),
      ardscales(blaze::subvector(theta, 1, theta.size()-1))
  {}

  template <typename VecType>
  inline
  Matern52::
  Matern52(double sigma_, VecType const& linescales_)
    : sigma(sigma_),
      ardscales(linescales_)
  {}

  inline blaze::DynamicVector<double>
  Matern52::
  vector() const
  {
    size_t n_dims = ardscales.size();
    auto theta    = blaze::DynamicVector<double>(1 + n_dims);
    theta[0]      = sigma;
    blaze::subvector(theta, 1, n_dims) = ardscales;
    return theta;
  }

  template<typename Kernel>
  inline blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>
  compute_gram_matrix(Kernel const& k,
		      blaze::DynamicMatrix<double> const& datamatrix);

  template<typename VecLHSType,
	   typename VecRHSType>
  inline double
  Matern52::
  operator()(VecLHSType const& x,
	     VecRHSType const& y) const noexcept
  {
    /*
     * Matern 5/2 kernel
     * \sigma * (1 + \sqrt{5}*r + 5/3*r^2) exp(-\sqrt{5}r)
     */
    auto r      = blaze::norm((x - y) / this->ardscales);
    auto s      = sqrt(5)*r;
    auto sigma2 = this->sigma * this->sigma;
    return sigma2 * (1 + s + s*s/3) * exp(-s);
  }

  template<typename Kernel>
  inline blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>
  compute_gram_matrix(Kernel const& k,
		      blaze::DynamicMatrix<double> const& datamatrix)
  {
    size_t n_data = datamatrix.columns();
    auto gram     = blaze::SymmetricMatrix<
      blaze::DynamicMatrix<double>>(n_data);
    for (size_t i = 0; i < n_data; ++i)
    {
      for (size_t j = i; j < n_data; ++j)
      {
	auto const& xi = blaze::column(datamatrix, i);
	auto const& xj = blaze::column(datamatrix, j);
	gram(i, j)     = k(xi, xj);
      }
    }
    return gram;
  }
}

#endif
