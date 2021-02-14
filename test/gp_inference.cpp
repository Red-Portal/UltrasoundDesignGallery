
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

#include "../src/gp/gp_prior.hpp"
#include "../src/gp/kernel.hpp"
#include "../src/inference/ess.hpp"
#include "../src/misc/mvnormal.hpp"
#include "../src/misc/prng.hpp"

#include <blaze/math/Subvector.h>

#include <limits>

template <typename Rng>
inline blaze::DynamicMatrix<double>
generate_mvsamples(Rng& rng, size_t n_dims, size_t n_points)
{
  auto data = blaze::DynamicMatrix<double>(n_dims, n_points);
  for (size_t i = 0; i < n_points; ++i)
  {
    blaze::column(data, i) = usvg::rmvnormal(rng, n_dims);
  }
  return data;
}

TEST_CASE("Identifiability check of GP hyperparameters using ESS", "[gp & ess]")
{
  auto linescales = blaze::DynamicVector<double>({1.0, 2.0, 3.0});
  auto sigma      = 0.6;
  auto kernel     = usvg::Matern52{sigma, linescales};
  auto prng       = usvg::Random123();

  size_t n_points = 128;
  size_t n_dims   = linescales.size();

  auto data   = generate_mvsamples(prng, n_dims, n_points);
  auto K      = usvg::compute_gram_matrix(kernel, data);
  auto K_chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( K_chol = usvg::cholesky_nothrow(K).value() );
  
  auto Z = usvg::rmvnormal(prng, n_points);
  auto y = K_chol.L * Z;
  
  auto mll = [&data, n_dims, n_points, &y](
    blaze::DynamicVector<double> const& x)->double{
    auto _sigma      = exp(x[0]);
    auto _linescales = exp(blaze::subvector(x, 1u, n_dims));
    auto _kernel     = usvg::Matern52{_sigma, _linescales};
    auto _K          = usvg::compute_gram_matrix(_kernel, data); 
    auto zero_mean   = blaze::zero<double>(n_points);
    if(auto _K_chol = usvg::cholesky_nothrow(_K))
      return usvg::dmvnormal(y, zero_mean, _K_chol.value(), true);
    else
      return std::numeric_limits<double>::min();
  };

  auto x_init = blaze::DynamicVector<double>(1 + n_dims, 0.0);
  auto p_init = mll(x_init);

  auto prior_mean = blaze::zero<double>(x_init.size());
  auto prior_var  = blaze::DynamicVector<double>(x_init.size(), 2.0);
  auto prior_chol = usvg::Cholesky<usvg::DiagonalChol>();
  REQUIRE_NOTHROW( prior_chol = usvg::cholesky_nothrow(prior_var).value() );
  auto prior_dist = usvg::MvNormal<usvg::DiagonalChol>(prior_mean, prior_chol);

  size_t n_samples = 1024;
  size_t n_burnin  = 128;
  auto samples     = blaze::DynamicMatrix<double>(1 + n_dims, n_samples);

  auto x = x_init;
  auto p = p_init;
  for (size_t i = 0; i < n_burnin; ++i)
  { /* burnin */
    auto [x_prop, p_prop, n_props] = usvg::ess_transition(
      prng, mll, x_init, p_init, prior_dist);
    x = x_prop;
    p = p_prop;
  }

  size_t n_total_props = 0;
  for (size_t i = 0; i < n_samples; ++i)
  {
    auto [x_prop, p_prop, n_props] = usvg::ess_transition(
      prng, mll, x, p, prior_dist);
    x = x_prop;
    p = p_prop;
    n_total_props += n_props;
    blaze::column(samples, i) = x;
  }

  auto sigma_est      = blaze::mean(exp(blaze::row(samples, 0)));
  REQUIRE(sigma_est == Approx(sigma).epsilon(0.1));

  size_t idx = 0;
  auto linescale_est = blaze::mean(exp(blaze::row(samples, idx+1)));
  REQUIRE(linescale_est == Approx(linescales[idx]).epsilon(0.2));

  ++idx;
  linescale_est = blaze::mean(exp(blaze::row(samples, idx+1)));
  REQUIRE(linescale_est == Approx(linescales[idx]).epsilon(0.2));

  ++idx;
  linescale_est = blaze::mean(exp(blaze::row(samples, idx+1)));
  REQUIRE(linescale_est == Approx(linescales[idx]).epsilon(0.2));
}
