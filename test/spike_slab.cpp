
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

#include <catch2/catch.hpp>
#define BLAZE_USE_DEBUG_MODE 1

#include "../src/inference/spike_slab_mcmc.hpp"
#include "../src/misc/prng.hpp"
#include "../src/misc/blaze.hpp"
#include "../src/misc/mvnormal.hpp"

#include <progressbar.hpp>

#include <random>

TEST_CASE("spike and slab proposals", "[spike slab]")
{
  auto key         = GENERATE(range(0u, 8u));
  auto prng        = usdg::Random123(key);
  double alpha     = 0.8;
  size_t n_samples = 1024;
  size_t n_burn    = 1024;
  size_t n_dims    = 8;
  size_t n_hyper   = 3;

  auto prior_mean = blaze::DynamicVector<double>( {0.0, 0.0, 0.0} );
  auto prior_var  = usdg::cholesky_nothrow(
    blaze::DynamicVector<double>( {1.0, 1.0, 1.0} ) ).value();
  auto prior_dist = usdg::MvNormal<usdg::DiagonalChol>{std::move(prior_mean),
    std::move(prior_var)};

  auto target_mean = blaze::DynamicVector<double>( {1.0, 1.0, 1.0} );
  auto target_var  = usdg::cholesky_nothrow(
    blaze::DynamicVector<double>( {0.1, 0.1, 0.1} )).value();
  auto target_dist = usdg::MvNormal<usdg::DiagonalChol>{std::move(target_mean),
    std::move(target_var)};

  auto prior = [&](blaze::DynamicVector<double> const& hyper) -> double{
    return prior_dist.logpdf(hyper);
  };

  auto target = [&](blaze::DynamicVector<double> const&,
		    blaze::DynamicVector<double> const& hyper) -> double {
    return target_dist.logpdf(hyper);
  };

  auto gamma_samples = blaze::DynamicMatrix<double>(n_dims,  n_samples);
  auto rho_samples   = blaze::DynamicMatrix<double>(n_dims,  n_samples);
  auto hyper_samples = blaze::DynamicMatrix<double>(n_hyper, n_samples);

  auto unif  = std::uniform_real_distribution<double>(0, 1);
  auto gamma = blaze::DynamicVector<double>(n_dims, 1.0);
  auto rho   = blaze::DynamicVector<double>(n_dims);
  for (size_t i = 0; i < n_dims; ++i)
  {
    rho[i] = unif(prng);
  }
  auto hyper = prior_dist.sample(prng);
  auto pm    = target(rho, hyper);

  auto pg        = progressbar(static_cast<int>(n_samples+n_burn));
  double acc_sum = 0.0;
  for (size_t i = 0; i < n_samples + n_burn; ++i)
  {
    pg.update();
    auto [gamma_acc, rho_acc, hyper_acc, pm_acc, acc] = usdg::spike_slab_mcmc(
      prng, target, gamma, rho, hyper, pm, alpha, prior);

    gamma = gamma_acc;
    rho   = rho_acc;
    hyper = hyper_acc;
    pm    = pm_acc;

    acc_sum += acc;

    if(i > n_burn)
    {
      blaze::column(gamma_samples, i-n_burn) = gamma;
      blaze::column(rho_samples,   i-n_burn) = rho;
      blaze::column(hyper_samples, i-n_burn) = hyper;
    }
  }
  std::cout << std::endl;

  for (size_t i = 0; i < 3; ++i)
  {
    auto mu    = blaze::mean(blaze::row(hyper_samples, i));
    auto sigma = blaze::stddev(blaze::row(hyper_samples, i));
    double z   = (mu - target_dist.mean[i]) / sigma;
    REQUIRE(abs(z) < 3);
  }

  std::cout << "acceptance: " << acc_sum/static_cast<double>(n_samples+n_burn)
	    << std::endl;
}


