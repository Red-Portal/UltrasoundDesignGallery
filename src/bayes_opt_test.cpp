
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
#include "system/profile.hpp"

#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/algorithms/nlopt.hpp>
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
    auto x = (lb/2) + dx*static_cast<double>(i);
    auto y = f(x);
    if(y >= y_opt)
    {
      y_opt = y;
      x_opt = x;
    }
  }
  return x_opt;
}

template <typename Acq, typename Func, typename BudgetType>
void
bayesian_optimization(usdg::Random123& prng,
		      Func objective,
		      size_t n_dims,
		      size_t n_init,
		      size_t n_burn,
		      size_t n_samples,
		      size_t n_iter,
		      BudgetType budget,
		      size_t n_pseudo,
		      double sigma,
		      spdlog::logger* logger)
{
  size_t n_params = 4;
  auto prior_mean = blaze::DynamicVector<double>(n_params, -1.0);
  auto prior_var  = blaze::DynamicVector<double>(n_params, 1.0);
  auto prior_chol = usdg::cholesky_nothrow(prior_var).value();
  auto prior_dist = usdg::MvNormal<usdg::DiagonalChol>{prior_mean, prior_chol};
  auto profiler   = usdg::Profiler{};

  double y_opt    = std::numeric_limits<double>::lowest();
  auto optimizer  = usdg::BayesianOptimization<Acq>(n_dims, n_pseudo);
  auto noise_dist = std::normal_distribution<double>(0.0, sigma);
  auto init_x     = optimizer.initial_queries(prng, n_init, logger);
  for (auto& [x, xi] : init_x)
  {
    auto noisy_objective = [&](double alpha_in) {
      return objective(x + alpha_in*xi) + noise_dist(prng);
    };
    auto [lb, ub] = usdg::pbo_find_bounds(x, xi);
    auto alpha    = naive_linesearch(noisy_objective, lb, ub, 1000);
    auto y        = objective(x + xi*alpha);
    auto betas    = usdg::sample_beta(prng, alpha, lb, ub, 0, n_pseudo, n_dims);

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
					 &profiler,
					 logger);

    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << objective(x) << std::endl;

    auto noisy_objective = [&](double alpha_in) {
      return objective(x + alpha_in*xi) + noise_dist(prng);
    };
    auto [lb, ub] = usdg::pbo_find_bounds(x, xi);
    auto alpha    = naive_linesearch(noisy_objective, lb, ub, 1000);
    auto y        = objective(x + xi*alpha);
    auto betas    = usdg::sample_beta(prng, alpha, ub, lb, iter, n_pseudo, n_dims);
    if (y > y_opt)
    {
      y_opt = y;
    }
    optimizer.push_data(x, xi, betas, alpha);
    if (logger)
    {
      logger->info("iter = {:>4}, y = {:g}, y opt = {:g}", iter, y, y_opt);
    }
    std::cout << profiler;
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
  return -res;
}

auto A = blaze::DynamicMatrix<double>({{10, 3, 17, 3.5, 1.7, 8},
				       {0.05, 10, 17, 0.1, 8, 14},
				       {3, 3.5, 1.7, 10, 17, 8},
				       {17, 8, 0.05, 10, 0.1, 14}});

auto P = blaze::DynamicMatrix<double>({{1312, 1696, 5569, 124, 8283, 5886},
				       {2329, 4135, 8307, 3736, 1004, 9991},
				       {2348, 1451, 3522, 2883, 3047, 6650},
				       {4047, 8828, 8732, 5743, 1091, 381}});
auto alpha = blaze::DynamicVector<double>({1.0, 1.2, 3.0, 3.2});

inline double
hartmann(blaze::DynamicVector<double> const& x_in)
{
  double res = 0.0;
  for (size_t i = 0; i < 4; ++i)
  {
    double local_sum = 0.0;
    for (size_t j = 0; j < 6; ++j)
    {
      local_sum += A(i,j) * pow(x_in[j] - 1e-4*P(i,j), 2);
    }
    res += alpha[i]*exp(-local_sum);
  }
  return res;
}

int main()
{
  auto key  = 0u;
  auto prng = usdg::Random123(key);

  auto console  = spdlog::stdout_color_mt("console");
  spdlog::set_level(spdlog::level::info);
  auto logger  = spdlog::get("console");

  size_t n_dims    = 6;
  size_t n_init    = 4;
  size_t n_burn    = 64;
  size_t n_samples = 32;
  size_t n_iter    = 50;
  size_t budget    = 1000;
  size_t n_pseudo  = 8;
  double sigma     = 0.0001;

  //std::cout << hartmann(blaze::DynamicVector<double>({0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573})) << std::endl;
  //usdg::render_function(hartmann);

  bayesian_optimization<usdg::ExpectedImprovementKoyama>(
  //bayesian_optimization<usdg::ExpectedImprovement>(
  //bayesian_optimization<usdg::ThompsonSampling>(
  //bayesian_optimization<usdg::ThompsonSamplingKoyama>(
    prng,
    hartmann,//rosenbrock,
    n_dims,					   
    n_init,
    n_burn,
    n_samples,
    n_iter,
    budget,
    n_pseudo,
    sigma,
    logger.get());

  //auto optimizer  = usdg::BayesianOptimization<>(
  //auto optimizer  = usdg::BayesianOptimization<usdg::ThompsonSampling>(
  //auto optimizer  = usdg::BayesianOptimization<usdg::ThompsonSamplingKoyama>(
}
