
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

#ifndef __US_GALLERY_DATAPOINT_HPP__
#define __US_GALLERY_DATAPOINT_HPP__

#include "gp_prior.hpp"

#include "../misc/blaze.hpp"

#include <cstdlib>
#include <utility>
#include <vector>
#include <cassert>

namespace usdg
{
  struct Datapoint
  {
    double alpha;
    blaze::DynamicVector<double> betas;
    blaze::DynamicVector<double> xi;
    blaze::DynamicVector<double> x;
  };

  class Dataset
  {
    size_t _n_dims;
    size_t _n_pseudo;
    std::vector<Datapoint> _data;

    // private:
    //   inline blaze::DynamicVector<double> const&
    //   pseudo_obs(size_t idx) const noexcept;

  public:
    inline Dataset(size_t n_dims, size_t n_pseudo);

    inline size_t
    alpha_index(size_t datapoint_idx) const noexcept;

    inline size_t
    beta_index(size_t datapoint_idx, size_t beta_idx) const noexcept;

    inline Dataset&
    push_back(Datapoint const& datapoint);

    inline blaze::DynamicMatrix<double>
    data_matrix() const noexcept;

    inline size_t
    dims() const noexcept;

    inline size_t
    num_pseudo() const noexcept;

    inline size_t
    num_data() const noexcept;
  };

  inline 
  Dataset::
  Dataset(size_t n_dims, size_t n_pseudo)
    : _n_dims(n_dims), _n_pseudo(n_pseudo)
  { }

  inline size_t
  Dataset::
  alpha_index(size_t datapoint_idx) const noexcept
  {
    size_t n_pseudo = _n_pseudo;
    return datapoint_idx * (1 + n_pseudo);
  }

  inline size_t
  Dataset::
  beta_index(size_t datapoint_idx, size_t beta_idx) const noexcept
  {
    return alpha_index(datapoint_idx) + beta_idx + 1;
  }

// inline blaze::DynamicVector<double> const&
// Dataset::
// pseudo_obs(size_t idx) const noexcept
//   /* Linear indexing of pseudo-observations */
// {
//   size_t n_data   = _data.size();
//   size_t n_pseudo = _n_pseudo;
//   size_t n_offset = n_data * (1 + n_pseudo);

//   size_t datapoint_idx = idx / n_offset;
//   size_t pseudoobs_idx = idx % n_offset;
//   return _data[datapoint_idx].betas[pseudoobs_idx];
// }

  inline blaze::DynamicMatrix<double>
  Dataset::
  data_matrix() const noexcept
  {
    size_t n_data   = _data.size();
    size_t n_dims   = _n_dims;
    size_t n_pseudo = _n_pseudo;
    size_t n_total  = (n_pseudo + 1) * n_data;
    auto res        = blaze::DynamicMatrix<double>(n_total, n_dims);

    for (size_t i = 0; i < n_data; ++i)
    {
      auto& cur        = _data[i];
      size_t alpha_idx = alpha_index(i);
      blaze::row(res, alpha_idx) = blaze::trans(cur.x + cur.alpha*cur.xi);

      for (size_t j = 0; j < n_pseudo; ++j)
      {
	size_t beta_idx = beta_index(i, j);
	blaze::row(res, beta_idx) = blaze::trans(cur.x + cur.betas[j]*cur.xi);
      }
    }
    return res;
  }

  inline usdg::Dataset&
  Dataset::
  push_back(Datapoint const& datapoint)
  {
    assert( datapoint.betas.size() == _n_pseudo );
    _data.push_back(datapoint);
    return *this;
  }

  inline size_t
  Dataset::
  dims() const noexcept
  {
    return _n_dims;
  }

  inline size_t
  Dataset::
  num_pseudo() const noexcept
  {
    return _n_pseudo;
  }

  inline size_t
  Dataset::
  num_data() const noexcept
  {
    return _data.size();
  }
}

#endif
