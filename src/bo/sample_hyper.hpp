
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

#ifndef __US_GALLERY_SAMPLEHYPER_HPP__
#define __US_GALLERY_SAMPLEHYPER_HPP__

#include "../gp/gp_prior.hpp"
#include "../gp/likelihood.hpp"
#include "../inference/pm_ess.hpp"
#include "../misc/blaze.hpp"

#include <vector>

namespace usdg
{
  template <typename Rng>
  std::tuple<blaze::DynamicMatrix<double>,
	     blaze::DynamicMatrix<double>,
	     std::vector<usdg::Cholesky<usdg::DenseChol>>>
  sample_gp_hyper(Rng& prng,
		  usdg::Dataset const& data,
		  blaze::DynamicMatrix<double> const& data_mat,
		  size_t n_burn,
		  size_t n_samples,
		  blaze::DynamicVector<double> const& theta_init,
		  usdg::MvNormal<usdg::DiagonalChol> const& prior_dist,
		  spdlog::logger* logger)
  {
    if(logger)
    {
      logger->info("Sampling from Gaussian process posterior: {}",
		   usdg::file_name(__FILE__));
    }

    double sigma_buf = 0.01;
    auto grad_hess = [&](blaze::DynamicVector<double> const& f_in)
      ->std::tuple<blaze::DynamicVector<double>,
		   blaze::DynamicMatrix<double>>
      {
	auto delta = usdg::pgp_delta(f_in, data, sigma_buf);
	return usdg::pgp_loglike_gradneghess(delta, data, sigma_buf);
      };

    auto loglike = [&](blaze::DynamicVector<double> const& f_in){
      auto delta = usdg::pgp_delta(f_in, data, sigma_buf);
      return usdg::pgp_loglike(delta);
    };

    auto I = blaze::IdentityMatrix<double>(data_mat.columns());
    auto make_gram = [&](blaze::DynamicVector<double> const& theta_in)
      ->blaze::DynamicMatrix<double>
      {
	auto kernel = usdg::Matern52Iso{exp(theta_in[0]), blaze::exp(theta_in[1])};
	auto K      = usdg::compute_gram_matrix(kernel, data_mat);
	sigma_buf   = exp(theta_in[2]);
	return K + (exp(theta_in[3])*I);
      };

    auto samples = usdg::pm_ess(prng,
				loglike,
				grad_hess,
				make_gram,
				theta_init,
				prior_dist,
				data_mat.columns(),
				n_samples,
				n_burn,
				logger);
    if(logger)
    {
      logger->info("Sampled from Gaussian process posterior.");
    }
    return samples;
  }
}

#endif
