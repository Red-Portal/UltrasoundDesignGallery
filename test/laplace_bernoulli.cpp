
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

#include "elliptical_slice.hpp"
#include "../src/gp/gp_prior.hpp"
#include "../src/gp/kernel.hpp"
#include "../src/inference/ess.hpp"
#include "../src/inference/laplace.hpp"
#include "../src/inference/pm_ess.hpp"
#include "../src/misc/mvnormal.hpp"
#include "../src/misc/prng.hpp"
#include "utils.hpp"

#include <blaze/math/Subvector.h>
#include <blaze/math/SymmetricMatrix.h>

#include <limits>
#include <cassert>

#include <spdlog/sinks/stdout_color_sinks.h>

inline double
sigmoid(double x)
{
  return 1 / (1 + exp(-x));
}

template<typename VecType>
inline blaze::DynamicVector<double>
sigmoid(VecType const& x)
{
  size_t n_elem = x.size();
  auto p        = blaze::DynamicVector<double>(n_elem);
  for (size_t i = 0; i < n_elem; ++i)
  {
    p[i] = 1 / (1 + exp(-x[i]));
  }
  return p;
}

inline double
binary_accuracy(blaze::DynamicVector<double> const& p,
		blaze::DynamicVector<double> const& y)
{
  assert(p.size() == y.size());
  size_t correct = 0;
  for (size_t i = 0; i < y.size(); ++i)
  {
    if (p[i] >= 0.5 && y[i] == 1.0)
    {
      ++correct;
    }
    else if (p[i] < 0.5 && y[i] == -1.0)
    {
      ++correct;
    }
  }
  return static_cast<double>(correct) / static_cast<double>(y.size()); 
}

TEST_CASE("laplace approximation of latent GP", "[laplace]")
{
  auto key        = 0u;//GENERATE(range(0u, 8u));
  auto prng       = usdg::Random123(key);
  size_t n_dims   = 4;
  size_t n_points = 50;

  auto kernel = usdg::Matern52ARD{1.0, blaze::DynamicVector<double>(n_dims, 0.5)};
  auto data_x = generate_mvsamples(prng, n_dims, n_points);
  auto K      = usdg::compute_gram_matrix(kernel, data_x);
  auto K_chol = usdg::Cholesky<usdg::DenseChol>();
  REQUIRE_NOTHROW( K_chol = usdg::cholesky_nothrow(K).value() );

  auto Z       = usdg::rmvnormal(prng, K.rows());
  auto f_truth = K_chol.L * Z;
  auto p       = sigmoid(f_truth);
  auto data_y  = blaze::map(p, [](double p_in)->double{
    return p_in >= 0.5 ? 1.0 : -1.0;
  });
  auto data_t = blaze::evaluate((blaze::eval(data_y) + 1)/2);

  auto grad_hess = [&data_t](blaze::DynamicVector<double> const& f_in)
    ->std::tuple<blaze::DynamicVector<double>,
		 blaze::DynamicMatrix<double>>
    {
      auto pi        = sigmoid(f_in);
      auto grad      = blaze::evaluate(data_t - pi);
      auto hess_diag = blaze::evaluate(pi * (pi - 1));
      auto n_hess    = hess_diag.size();
      auto neg_hess  = blaze::DynamicMatrix<double>(n_hess, n_hess, 0.0);
      for (size_t i = 0; i < n_hess; ++i)
      {
	neg_hess(i,i) = -hess_diag[i];
      }
      return {std::move(grad), std::move(neg_hess)};
    };


  auto console  = spdlog::stdout_color_mt("console");
  spdlog::set_level(spdlog::level::info);
  auto logger  = spdlog::get("console");

  auto loglike = [&data_y](blaze::DynamicVector<double> const& f_in)->double{
    return -1*blaze::sum(blaze::log(1 + blaze::exp(-1*data_y*f_in)));
  };

  auto res = usdg::laplace_approximation(K_chol,
					 f_truth.size(),
					 grad_hess,
					 loglike,
					 20,
					 logger.get());
  auto res_tup = std::tuple<blaze::DynamicVector<double>,
			    blaze::SymmetricMatrix<blaze::DynamicMatrix<double>>>();
  REQUIRE_NOTHROW(res_tup = res.value());
  auto [f, W] = res_tup;
  auto acc    = binary_accuracy(sigmoid(f), data_y);
  REQUIRE(acc >= 0.9);
}
