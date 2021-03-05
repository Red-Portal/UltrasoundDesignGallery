
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

#ifndef __US_GALLERY_CMAES_HPP__
#define __US_GALLERY_CMAES_HPP__

#include "../misc/blaze.hpp"
#include "../misc/debug.hpp"

#include <vector>

#include <pagmo/algorithms/cmaes.hpp>

namespace usdg
{
  template <typename Acq>
  struct BoundedProblem {
    Acq acq;
    std::vector<double> lb;
    std::vector<double> ub;

    inline std::vector<double>
    fitness(std::vector<double> const& x) const
    {
      auto x_in = blaze::DynamicVector<double>(x.begin(), x.end());
      return { acq(x_in) };
    }

    inline std::pair<std::vector<double>,
		     std::vector<double>>
    get_bounds() const
    {
      return { lb, ub };
    }
  };

  template <typename Rng,
	    typename ObjectiveFunc>
  inline std::pair<blaze::DynamicVector<double>, double>
  cmaes_optimize(Rng& prng,
		 ObjectiveFunc objective,
		 size_t n_dims,
		 size_t budget,
		 spdlog::logger* logger)
  {
    double sigma0  = sqrt(n_dims)/4;
    double ftol    = 1e-6;
    double xtol    = 1e-3;
    size_t n_pop   = 4 + static_cast<size_t>(
      ceil(3*log(static_cast<double>(n_dims))));
    auto unif_dist = std::uniform_real_distribution<double>(0, 1);
    auto prob      = pagmo::problem{
      usdg::BoundedProblem<ObjectiveFunc>{
	objective,
	std::vector<double>(n_dims, 0.0),
	std::vector<double>(n_dims, 1.0)}
    };
    auto pop = pagmo::population{prob};
    for (size_t i = 0; i < n_pop; ++i)
    {
      auto vec = std::vector<double>(n_dims);
      for (size_t j = 0; j < n_dims; ++j)
      {
	vec[j] = unif_dist(prng);
      }
      pop.push_back(std::move(vec));
    }

    unsigned int budg_tmp = static_cast<unsigned int>(budget);
    auto user_algo = pagmo::cmaes{budg_tmp, -1, -1, -1, -1,
      sigma0, ftol, xtol, false, true};
    user_algo.set_verbosity(1u);
    pop = user_algo.evolve(pop);

    auto champ_x = blaze::DynamicVector<double>(
      pop.get_pupluation().champion_x());
    auto champ_f = pop.get_pupluation().champion_f()[0];
    return {std::move(champ_x), champ_f};
  }
}

#endif
