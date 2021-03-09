
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

#include "bo/acquisition.hpp"
#include "bo/bayesian_optimization.hpp"
#include "bo/find_bounds.hpp"
#include "gp/sample_beta.hpp"
#include "math/mvnormal.hpp"
#include "math/prng.hpp"
#include "math/uniform.hpp"

#include <pagmo/algorithms/cmaes.hpp>
#include <matplotlib-cpp/matplotlibcpp.h>
#include <spdlog/sinks/stdout_color_sinks.h>

template <typename Func>
double
naive_linesearch(Func f, double lb, double ub, size_t resol)
{
  double dx    = (ub - lb)/static_cast<double>(resol);
  double y_opt = std::numeric_limits<double>::lowest();
  double x_opt = 0.0;
  for (size_t i = 0; i <= resol; ++i)
  {
    auto x = lb + dx*static_cast<double>(i);
    auto y = f(x);
    if(y >= y_opt)
    {
      y_opt = y;
      x_opt = x;
    }
  }
  return x_opt;
}

template <typename Func>
void
bayesian_optimization(usdg::Random123& prng,
		      Func objective,
		      spdlog::logger* logger)
{
  size_t n_dims    = 10;
  size_t n_init    = 8;
  size_t n_burn    = 64;
  size_t n_samples = 64;
  size_t n_iter    = 20;
  size_t budget    = 1000;
  size_t n_pseudo  = 4;
  double sigma     = 0.01;

  size_t n_params = 4;
  auto prior_mean = blaze::DynamicVector<double>(n_params, -1.0);
  auto prior_var  = blaze::DynamicVector<double>(n_params, 1.0);
  auto prior_chol = usdg::cholesky_nothrow(prior_var).value();
  auto prior_dist = usdg::MvNormal<usdg::DiagonalChol>{prior_mean, prior_chol};

  double y_opt    = std::numeric_limits<double>::lowest();
  //auto optimizer  = usdg::BayesianOptimization<usdg::ExpectedImprovement>(
  //auto optimizer  = usdg::BayesianOptimization<usdg::ThompsonSampling>(
  //auto optimizer  = usdg::BayesianOptimization<usdg::ExpectedImprovementKoyama>(
  auto optimizer  = usdg::BayesianOptimization<usdg::ThompsonSamplingKoyama>(
      n_dims, n_pseudo);
  auto noise_dist = std::normal_distribution<double>(0.0, sigma);
  auto init_x     = optimizer.initial_queries(prng, n_init, logger);
  for (auto& [x, xi] : init_x)
  {
    auto noisy_objective = [&](double alpha) {
      return objective(x + alpha*xi) + noise_dist(prng);
    };
    auto [lb, ub] = usdg::pbo_find_bounds(x, xi);
    auto alpha    = naive_linesearch(noisy_objective, lb, ub, 100);
    auto y        = objective(x + xi*alpha);
    auto betas    = usdg::sample_beta(prng, alpha, lb, ub, 0, n_samples, n_dims);

    if (y > y_opt)
    {
      y_opt = y;
    }
    optimizer.push_data(x, xi, betas, alpha);
  }
  if (logger)
  {
    logger->info("iter = {:>4}, y opt = {:g}", 0, y_opt);
  }

  for (size_t iter = 1; iter < n_iter; ++iter)
  {
    auto [x, xi]  = optimizer.next_query(prng,
					 n_burn,
					 n_samples,
					 budget,
					 //prev_theta,
					 prior_dist,
					 logger);

    auto noisy_objective = [&](double alpha) {
      return objective(x + alpha*xi) + noise_dist(prng);
    };
    auto [lb, ub] = usdg::pbo_find_bounds(x, xi);
    auto alpha    = naive_linesearch(noisy_objective, lb, ub, 100);
    auto y        = objective(x + xi*alpha);
    auto betas    = usdg::sample_beta(prng, alpha, ub, lb, iter, n_samples, n_dims);
    if (y > y_opt)
    {
      y_opt = y;
    }
    optimizer.push_data(x, xi, betas, alpha);
    if (logger)
    {
      logger->info("iter = {:>4}, y opt = {:g}", iter, y_opt);
    }
  }
}

inline double
rosenbrock(blaze::DynamicVector<double> const& x_in)
{
  auto x = blaze::evaluate(2.048*(x_in*2 - 1.0));
  size_t n_dims = x.size();
  double res    = 0.0;
  for (size_t i = 0; i < n_dims-1; ++i)
  {
    double t1 = x[i+1] - (x[i]*x[i]);
    double t2 = (x[i] - 1);
    res      += 100*(t1*t1) + t2*t2;
  }

  if (std::isnan(res))
  {
    throw std::runtime_error("nan");
  }
  return -res;
}

int main()
{
  auto key  = 0u;
  auto prng = usdg::Random123(key);

  auto console  = spdlog::stdout_color_mt("console");
  spdlog::set_level(spdlog::level::info);
  auto logger  = spdlog::get("console");

  bayesian_optimization(prng, rosenbrock, logger.get());
}
