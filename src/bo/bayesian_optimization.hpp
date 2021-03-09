
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

#ifndef __US_GALLERY_BAYESOPT_HPP__
#define __US_GALLERY_BAYESOPT_HPP__

#include "../bo/find_bounds.hpp"

#include "../gp/data.hpp"
#include "../math/blaze.hpp"
#include "../math/debug.hpp"

namespace usdg
{ 
  template <typename AcqFunc>
  struct BayesianOptimization
  {
    size_t _n_dims;
    usdg::Dataset _data;
//    blaze::DynamicVector<double> _x_opt;

  public:
    inline
    BayesianOptimization(size_t n_dims, size_t n_pseudo);

    template <typename Rng>
    inline std::vector<
      std::pair<blaze::DynamicVector<double>,
		blaze::DynamicVector<double>>>
    initial_queries(Rng& prng, size_t n_init, spdlog::logger* logger) const;
    
    template <typename Rng>
    inline std::tuple<blaze::DynamicVector<double>,
		      blaze::DynamicVector<double>>//,
			//blaze::DynamicMatrix<double>>
    next_query(Rng& prng,
	       size_t n_burn,
	       size_t n_samples,
	       size_t budget,
	       usdg::MvNormal<usdg::DiagonalChol> const& prior_dist,
	       spdlog::logger* logger) const;

    inline void
    push_data(blaze::DynamicVector<double> const& x,
	      blaze::DynamicVector<double> const& xi,
	      blaze::DynamicVector<double> const& betas,
	      double alpha);
  };

  template <typename AcqFunc>
  inline
  BayesianOptimization<AcqFunc>::
  BayesianOptimization(size_t n_dims, size_t n_pseudo)
    : _n_dims(n_dims),
      _data(n_dims, n_pseudo)
  { }

  template <typename AcqFunc>
  template <typename Rng>
  inline std::vector<
    std::pair<blaze::DynamicVector<double>,
	      blaze::DynamicVector<double>>>
  BayesianOptimization<AcqFunc>::
  initial_queries(Rng& prng, size_t n_init, spdlog::logger* logger) const
  {
    if (logger)
    {
      logger->info("Generating {} initial points.", n_init);
    }

    auto res     = std::vector<std::pair<blaze::DynamicVector<double>,
					 blaze::DynamicVector<double>>>();
    auto dist    = std::uniform_real_distribution<double>(0, 1);
    auto vecdist = blaze::generate(
      this->_n_dims, [&prng, &dist](size_t)->double { return dist(prng); });
    res.reserve(n_init);
    for (size_t i = 0; i < n_init; ++i)
    {
      auto x  = blaze::DynamicVector<double>(vecdist);
      auto xi = usdg::rmvnormal(prng, this->_n_dims);
      xi     /= blaze::max(blaze::abs(xi));
      res.emplace_back(std::make_pair(std::move(x), std::move(xi)));
    }

    if (logger)
    {
      logger->info("Generated initial points.");
    }
    return res;
  }

  template <typename AcqFunc>
  inline void
  BayesianOptimization<AcqFunc>::
  push_data(blaze::DynamicVector<double> const& x,
	    blaze::DynamicVector<double> const& xi,
	    blaze::DynamicVector<double> const& betas,
	    double alpha)
  {

    auto point = usdg::Datapoint{alpha, betas, xi, x};
    this->_data.push_back(std::move(point));
  }
}

#endif
