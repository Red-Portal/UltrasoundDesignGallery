
#ifndef __US_GALLERY_KERNEL_HPP__
#define __US_GALLERY_KERNEL_HPP__

#include "gp_prior.hpp"

#include <blaze/math/DenseVector.h>
#include <blaze/math/DenseMatrix.h>

#include <cmath>

namespace gp
{
  struct Matern52
  {
    double sigma2;
    blaze::DynamicVector<double> ardscales;

    inline double operator()(blaze::DynamicVector<double> const& x,
			     blaze::DynamicVector<double> const& y) const noexcept;
  };

  template<typename Kernel>
  inline blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>
  compute_gram_matrix(Kernel const& k,
		      blaze::DynamicMatrix<double> const& datamatrix);
}

inline double
gp::Matern52::
operator()(blaze::DynamicVector<double> const& x,
	   blaze::DynamicVector<double> const& y) const noexcept
{
  /*
   * Matern 5/2 kernel
   * \sigma * (1 + \sqrt{5}*r + 5/3*r^2) exp(-\sqrt{5}r)
   */
  auto constexpr sqrt5 = sqrt(5);
  auto r = blaze::norm((x - y) / this->ardscales);
  auto s = sqrt5*r;
  return this->sigma2 * (1 + s + s*s/3) * exp(-s);
}

template<typename Kernel>
inline blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>
gp::
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


#endif
