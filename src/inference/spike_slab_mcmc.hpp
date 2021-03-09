
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

#ifndef __US_GALLERY_SPIKESLAB_HPP__
#define __US_GALLERY_SPIKESLAB_HPP__

#include "../misc/blaze.hpp"
#include "../misc/mvnormal.hpp"

#include <random>
#include <cmath>
#include <limits>

namespace usdg
{
  template <typename Rng>
  inline void
  add_move(Rng& prng,
	   blaze::DynamicVector<double>& gamma,
	   blaze::DynamicVector<double>& rho)
  {
    auto excluded   = 1 - gamma;
    auto gamma_dist = std::discrete_distribution<size_t>(excluded.begin(),
							 excluded.end());
    auto k        = gamma_dist(prng);
    gamma[k]      = 1.0;
    auto rho_dist = std::uniform_real_distribution<double>(0, 1);
    rho[k]        = rho_dist(prng);
  }

  template <typename Rng>
  inline void
  delete_move(Rng& prng,
	      blaze::DynamicVector<double>& gamma,
	      blaze::DynamicVector<double>& rho)
  {
    auto& included  = gamma;
    auto gamma_dist = std::discrete_distribution<size_t>(included.begin(),
							 included.end());
    auto k        = gamma_dist(prng);
    gamma[k]      = 0.0;
    rho[k]        = 1.0;
  }

  template <typename Rng>
  inline void
  swap_move(Rng& prng,
	    blaze::DynamicVector<double>& gamma,
	    blaze::DynamicVector<double>& rho)
  {
    add_move(prng, gamma, rho);
    delete_move(prng, gamma, rho);
  }

  template <typename Rng,
	    typename VecType>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>>
  spike_slab_proposal(Rng& prng,
		      VecType const& gamma_cur,
		      VecType const& rho_cur)
  {
    auto gamma_prop = blaze::DynamicVector<double>(gamma_cur);
    auto rho_prop   = blaze::DynamicVector<double>(rho_cur);

    auto dist = std::uniform_int_distribution<size_t>(0, 2);
    auto move = dist(prng);
  
    if(move == 0)
    {
      add_move(prng, gamma_prop, rho_prop);
    }
    else if(move == 1)
    {
      delete_move(prng, gamma_prop, rho_prop);
    }
    else if(move == 2)
    {
      swap_move(prng, gamma_prop, rho_prop);
    }
    return {std::move(gamma_prop), std::move(rho_prop)};
  }

  template <typename Rng>
  inline void
  rho_gibbs_move(Rng& prng,
		 blaze::DynamicVector<double>& gamma,
		 blaze::DynamicVector<double>& rho)
  {
    auto rho_dist = std::uniform_real_distribution<double>(0, 1);
    size_t n_dims = gamma.size();
    for (size_t i = 0; i < n_dims; ++i)
    {
      if(gamma == 1)
      {
	rho[i] = rho_dist(prng);
      }
    }
  }

  template <typename VecType>
  inline double
  bernoulli_jointpdf(VecType const& vec,
		     double alpha)
  {
    auto sum_x = blaze::evaluate(blaze::sum(vec));
    double n   = static_cast<double>(vec.size());
    return log(alpha) * sum_x + log(1 - alpha) * (n - sum_x);
  }

  template <typename Rng,
	    typename GammaType,
	    typename RhoType,
	    typename LoglikeFunc>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    double>
  spike_slab_mcmc(Rng& prng,
		  LoglikeFunc target,
		  GammaType const& gamma,
		  RhoType   const& rho,
		  double pm_prev,
		  double gamma_alpha)
  /* 
   * Markov-chain Monte Carlo with Gibbs move for spike-and-slab prior
   * note: alpha in p(gamma) = Bernoulli(alpha) is prespecified for simplicity
   * 
   * Slice sampling version of:
   * Variable Selection for Nonparametric Gaussian Process Priors: Models and Computational Strategies.
   * Terrance Savitsky, Marina Vannucci and Naijun Sha
   * Statistical Science, 2011.
   */
  {
    auto gamma_acc  = blaze::DynamicVector<double>(gamma.size());
    auto rho_acc    = blaze::DynamicVector<double>(rho.size());
    auto u_dist     = std::uniform_real_distribution<double>(0, 1);
    auto [gamma_prop, rho_prop] = spike_slab_proposal(prng, gamma, rho);

    auto gamma_prior_cur  = bernoulli_jointpdf(gamma,      gamma_alpha);
    auto gamma_prior_prop = bernoulli_jointpdf(gamma_prop, gamma_alpha);

    /* Metropolis-Hastings update of spike-and-slab parameters */
    double pm_prop    = target(rho_prop);
    double like_ratio = pm_prop - pm_prev + gamma_prior_cur - gamma_prior_prop;
    double mh_ratio   = std::min(0.0, like_ratio);
    if(log(u_dist(prng)) < mh_ratio)
    {
      return {std::move(gamma_prop), std::move(rho_prop), pm_prop};
    }
    else
    {
      return {std::move(gamma), std::move(rho), pm_prev};
    }
  }
}

#endif
