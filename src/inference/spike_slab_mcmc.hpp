
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
    else if(move == 1)
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
	    typename LoglikeFunc,
	    typename PriorFunc>
  inline std::tuple<blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    blaze::DynamicVector<double>,
		    double,
		    double>
  spike_slab_mcmc(Rng& prng,
		  LoglikeFunc target,
		  blaze::DynamicVector<double> const& gamma,
		  blaze::DynamicVector<double> const& rho,
		  blaze::DynamicVector<double> const& hyper,
		  double pm_prev,
		  double gamma_alpha,
		  PriorFunc prior)
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
    double pm         = std::numeric_limits<double>::lowest();
    double pm_prop    = target(gamma_prop, rho_prop, hyper);
    double like_ratio = pm_prop - pm_prev + gamma_prior_cur - gamma_prior_prop;
    double mh_ratio   = std::min(0.0, like_ratio);
    if(log(u_dist(prng)) < mh_ratio)
    {
      gamma_acc = std::move(gamma_prop);
      rho_acc   = std::move(rho_prop);
      pm        = pm_prop;
    }
    else
    {
      gamma_acc = gamma_prop;
      rho_acc   = rho_prop;
      pm        = pm_prev;
    }

    /* Slice-sampling update of other hyperparameters */
    double win_len  = 2.0;
    auto prior_cur  = prior(hyper);
    auto logy       = (pm + prior_cur) + log(u_dist(prng));
    auto n_hypers   = hyper.size();
    auto ls         = blaze::DynamicVector<double>(n_hypers);
    auto rs         = blaze::DynamicVector<double>(n_hypers);
    auto hyper_prop = blaze::DynamicVector<double>(n_hypers);

    for (size_t i = 0; i < n_hypers; ++i)
    {
      auto win_shrink = u_dist(prng);
      ls[i]           = hyper[i] - win_shrink*win_len;
      rs[i]           = ls[i] + win_len;
    }

    size_t rejections = 0; 
    while(true)
    {
      for (size_t i = 0; i < n_hypers; ++i)
      {
	auto u        = u_dist(prng);
	hyper_prop[i] = ls[i] + u*(rs[i] - ls[i]);
      }
      //std::cout << blaze::trans(hyper_prop) << std::endl;

      pm_prop = target(gamma_acc, rho_acc, hyper_prop);
      if (logy < pm_prop + prior(hyper_prop))
      {
	pm = pm_prop;
	break;
      }
      else
      {
	++rejections;
	for (size_t i = 0; i < n_hypers; ++i)
	{
	  if(hyper_prop[i] < hyper[i])
	  {
	    ls[i] = hyper_prop[i];
	  }
	  else
	  {
	    rs[i] = hyper_prop[i];
	  }
	}
      }
    }
    return {std::move(gamma_acc),
      std::move(rho_acc),
      std::move(hyper_prop),
      pm_prop,
      static_cast<double>(1)/static_cast<double>(rejections+1)};
  }
}

#endif
