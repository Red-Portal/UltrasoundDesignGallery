
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

#include "../math/blaze.hpp"

#include <cmath>
#include <iostream>

namespace usdg
{
  struct Matern52Iso
  {
    double sigma;
    double scale;

    inline Matern52Iso();

    template <typename VecType>
    inline Matern52Iso(VecType const& theta);
    
    inline Matern52Iso(double sigma, double scale);

    template<typename VecLHSType,
	     typename VecRHSType>
    inline double operator()(VecLHSType const& x,
			     VecRHSType const& y) const noexcept;

    inline blaze::DynamicVector<double> vector() const;
  };

  inline
  Matern52Iso::
  Matern52Iso()
    : sigma(), scale()
  {}

  template <typename VecType>
  inline
  Matern52Iso::
  Matern52Iso(VecType const& theta)
    : sigma(theta[0]), scale(theta[1])
  {}

  inline
  Matern52Iso::
  Matern52Iso(double sigma_, double scale_)
    : sigma(sigma_), scale(scale_)
  {}

  inline blaze::DynamicVector<double>
  Matern52Iso::
  vector() const
  {
    auto theta    = blaze::DynamicVector<double>(2);
    theta[0]      = sigma;
    theta[1]      = scale;
    return theta;
  }

  template<typename VecLHSType,
	   typename VecRHSType>
  inline double
  Matern52Iso::
  operator()(VecLHSType const& x,
	     VecRHSType const& y) const noexcept
  {
    /*
     * Matern 5/2 kernel
     * \sigma * (1 + \sqrt{5}*r + 5/3*r^2) exp(-\sqrt{5}r)
     */
    auto r      = blaze::norm((x - y) / this->scale);
    auto s      = sqrt(5)*r;
    auto sigma2 = this->sigma * this->sigma;
    return sigma2 * (1 + s + s*s/3) * exp(-s);
  }

  struct Matern52ARD
  {
    double sigma;
    blaze::DynamicVector<double> ardscales;

    inline Matern52ARD();

    template <typename VecType>
    inline Matern52ARD(VecType const& theta);
    
    template <typename VecType>
    inline Matern52ARD(double sigma, VecType const& linescales);

    template<typename VecLHSType,
	     typename VecRHSType>
    inline double operator()(VecLHSType const& x,
			     VecRHSType const& y) const noexcept;

    inline blaze::DynamicVector<double> vector() const;
  };

  inline
  Matern52ARD::
  Matern52ARD()
    : sigma(), ardscales()
  {}

  template <typename VecType>
  inline
  Matern52ARD::
  Matern52ARD(VecType const& theta)
    : sigma(theta[0]),
      ardscales(blaze::subvector(theta, 1, theta.size()-1))
  {}

  template <typename VecType>
  inline
  Matern52ARD::
  Matern52ARD(double sigma_, VecType const& linescales_)
    : sigma(sigma_),
      ardscales(linescales_)
  {}

  inline blaze::DynamicVector<double>
  Matern52ARD::
  vector() const
  {
    size_t n_dims = ardscales.size();
    auto theta    = blaze::DynamicVector<double>(1 + n_dims);
    theta[0]      = sigma;
    blaze::subvector(theta, 1, n_dims) = ardscales;
    return theta;
  }

  template<typename VecLHSType,
	   typename VecRHSType>
  inline double
  Matern52ARD::
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

  template <typename Kernel>
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

  template <typename XVecType,
	    typename YVecType>
  inline decltype(auto)
  derivative(Matern52Iso const& kernel,
	     XVecType const& dx,
	     YVecType const& y)
  {
    auto delta  = (dx - y) / kernel.scale;
    auto r      = blaze::norm(delta);
    auto sqrt5  = sqrt(5);
    auto sigma2 = kernel.sigma * kernel.sigma;
    auto s      = sqrt5*r;
    auto dsdx   = sqrt5*delta/kernel.scale/r;

    return (sigma2/-3.0*s*(1 + s)*exp(-s)) * dsdx;
  }
}

#endif
