
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

#include <csv.hpp>
#include <progressbar.hpp>
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
std::pair<std::vector<double>,
	  std::vector<double>>
bayesian_optimization(usdg::Random123& prng,
		      Func objective,
		      size_t n_dims,
		      size_t n_init,
		      size_t n_iter,
		      BudgetType budget,
		      size_t n_pseudo,
		      double sigma,
		      blaze::DynamicVector<double> linescales,
		      spdlog::logger* logger)
{
  double eps      = 1e-2;
  size_t n_params = 4;
  auto prior_mean = blaze::DynamicVector<double>(n_params, -1.0);
  auto prior_var  = blaze::DynamicVector<double>(n_params, 1.0);
  auto prior_chol = usdg::cholesky_nothrow(prior_var).value();
  auto prior_dist = usdg::MvNormal<usdg::DiagonalChol>{prior_mean, prior_chol};
  auto profiler   = usdg::Profiler{};

  auto optimizer  = usdg::BayesianOptimization<Acq>(n_dims, n_pseudo);
  auto noise_dist = std::normal_distribution<double>(0.0, sigma);
  auto init_x     = optimizer.initial_queries(prng, n_init, logger);
  auto start      = usdg::clock::now();
  auto hist_y_opt = std::vector<double>();
  auto hist_time  = std::vector<double>();
  for (auto& [x, xi] : init_x)
  {
    auto noisy_objective = [&, x=x, xi=xi](double alpha_in) {
      return objective(x + alpha_in*xi) + noise_dist(prng);
    };
    auto [lb, ub] = usdg::pbo_find_bounds(x, xi);
    auto alpha    = naive_linesearch(noisy_objective, lb, ub, 1000);
    auto betas    = blaze::DynamicVector<double>(n_pseudo);
    blaze::subvector(betas, 2, n_pseudo-2) =  usdg::sample_beta(prng, alpha,
								lb + eps,
								ub - eps,
								0, n_pseudo-2,
								n_dims);
    betas[0] = lb;
    betas[1] = ub;
    optimizer.push_data(x, xi, betas, alpha);
  }
  if (logger)
  {
    logger->info("iter = {:>4}", 0);
  }

  double y_opt = std::numeric_limits<double>::lowest();
  for (size_t iter = 1; iter < n_iter; ++iter)
  {
    auto [x, xi, x_opt, _]  = optimizer.next_query(prng, iter, n_pseudo,
						   budget, linescales,
						   &profiler, logger);

    auto noisy_objective = [&, x=x, xi=xi](double alpha_in) {
      return objective(x + alpha_in*xi) + noise_dist(prng);
    };
    auto [lb, ub] = usdg::pbo_find_bounds(x, xi);

    auto alpha    = naive_linesearch(noisy_objective, lb, ub, 1000);
    auto y        = objective(x + xi*alpha);
    auto betas    = blaze::DynamicVector<double>(n_pseudo);

    blaze::subvector(betas, 2, n_pseudo-2) = usdg::sample_beta(prng, alpha,
							       lb + eps,
							       ub - eps,
							       iter,
							       n_pseudo-2,
							       n_dims);
    betas[0] = lb;
    betas[1] = ub;

    y_opt = std::max(objective(x_opt), y_opt);
    optimizer.push_data(x, xi, betas, alpha);

    auto elapsed = usdg::compute_duration(start);
    hist_y_opt.push_back(y_opt);
    hist_time.push_back(elapsed.count());
    if (logger)
    {
      logger->info("iter = {:>4}, time = {:.2f}, y = {:g}, y opt = {:g}",
		   iter, elapsed.count(), y, y_opt);
    }
    //std::cout << profiler;
  }
  return { std::move(hist_y_opt), std::move(hist_time) };
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

inline double
hartmann(blaze::DynamicVector<double> const& x_in)
{
  auto const A = blaze::DynamicMatrix<double>({{  10,   3,   17, 3.5, 1.7,  8},
					       {0.05,  10,   17, 0.1,   8, 14},
					       {   3, 3.5,  1.7,  10,  17,  8},
					       {  17,   8, 0.05,  10, 0.1, 14}});

  auto const P = blaze::DynamicMatrix<double>({{1312, 1696, 5569, 124,  8283, 5886},
					       {2329, 4135, 8307, 3736, 1004, 9991},
					       {2348, 1451, 3522, 2883, 3047, 6650},
					       {4047, 8828, 8732, 5743, 1091,  381}});
  auto const alpha = blaze::DynamicVector<double>({1.0, 1.2, 3.0, 3.2});


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

inline double
ackley(blaze::DynamicVector<double> const& x_in)
{
  double a   = 20.0;
  double b   = 0.2;
  double c   = 2 * 3.141592;
  size_t d   = x_in.size();

  auto x = (x_in - 0.5)*2*32;
  
  auto t1    = -a * exp( -b * sqrt( blaze::dot(x, x)/static_cast<double>(d) ) );
  double res = 0.0;
  for (size_t i = 0; i < d; ++i)
  {
    res += cos(c * x[i]);
  }
  auto t2 = -exp(res / static_cast<double>(d));
  return -(t1 + t2 + a + exp(1));
}

inline double
stablinskytang(blaze::DynamicVector<double> const& x_in)
{
  auto x = 10*x_in - 5;
  return -blaze::sum(
    0.5*(blaze::pow(x, 4) - 17*blaze::pow(x, 2) + 5*x));
}

template <typename Strategy,
	  typename ObjFunc>
void
run_benchmark(std::string const& fname,
	      ObjFunc objective,
	      size_t n_dims,
	      spdlog::logger* logger)
{
  logger->info("Running benchmark for {}", fname);

  size_t n_reps   = 100;
  size_t n_init   = 4;
  size_t n_iter   = 50;
  size_t budget   = 10000;
  size_t n_pseudo = 16;
  double sigma    = 0.001;
  auto linescales = blaze::DynamicVector<double>(n_dims, 0.2);

  auto pb     = progressbar(static_cast<int>(n_reps)); 
  auto stream = std::ofstream(fname);
  auto writer = csv::make_csv_writer(stream);
  auto start  = usdg::clock::now();
#pragma omp parallel for
  for (size_t i = 0; i < n_reps; ++i)
  {
#pragma omp critical
    pb.update();
    auto prng = usdg::Random123(i);
    auto [hist_y, hist_t] = bayesian_optimization<Strategy>(
      prng, objective,  n_dims, n_init, n_iter, budget,
      n_pseudo, sigma, linescales, nullptr);

    auto& hist_y_local = hist_y;
#pragma omp critical
    writer << hist_y_local;
  }
  std::cout << std::endl;
  auto elapsed = usdg::compute_duration(start);
  logger->info("Finished running benchmark for {:.2f} seconds", elapsed.count());
}

template <typename ObjFunc>
void
run_randomsearch(std::string const& fname,
		 ObjFunc objective,
		 size_t n_dims)
{
  size_t n_reps   = 100;
  size_t n_init   = 4;
  size_t n_iter   = 50;
  double sigma    = 0.001;

  auto pb     = progressbar(static_cast<int>(n_reps)); 
  auto stream = std::ofstream(fname);
  auto writer = csv::make_csv_writer(stream);
  for (size_t i = 0; i < n_reps; ++i)
  {
    pb.update();
    auto prng       = usdg::Random123(i);
    double y_opt    = std::numeric_limits<double>::lowest();
    auto hist_y_opt = std::vector<double>();
    for (size_t t = 0; t < n_iter+n_init; ++t)
    {
      auto x  = usdg::rmvuniform(prng, n_dims, 0, 1);
      auto xi = usdg::rmvnormal(prng, n_dims);
      xi     /= blaze::max(blaze::abs(xi));

      auto noise_dist = std::normal_distribution<double>(0.0, sigma);
      auto noisy_objective = [&](double alpha_in) {
	return objective(x + alpha_in*xi) + noise_dist(prng);
      };

      auto [lb, ub] = usdg::pbo_find_bounds(x, xi);
      auto alpha    = naive_linesearch(noisy_objective, lb, ub, 1000);

      double y = objective(x + alpha*xi);
      y_opt    = std::max(y, y_opt);

      if(t > n_init)
      {
	hist_y_opt.push_back(y_opt);
      }
    }
    writer << hist_y_opt;
  }
}

int main()
{
  using namespace std::string_literals;

  auto console  = spdlog::stdout_color_mt("console");
  spdlog::set_level(spdlog::level::info);
  auto logger  = spdlog::get("console");

  // auto _objective      = rosenbrock;
  // auto _objective_name = "rosenbrock10D"s;
  // run_benchmark<usdg::AEI_AEI>( _objective_name + "_AEI_AEI.csv"s,   _objective, 10, logger.get());

  // auto _objective      = ackley;
  // auto _objective_name = "ackley20D"s;
  // run_benchmark<usdg::AEI_AEI>( _objective_name + "_AEI_AEI.csv"s,  _objective, 20, logger.get());
  // exit(1);

  {
    auto objective      = rosenbrock;
    size_t n_dims       = 10;
    auto objective_name = "rosenbrock10D"s;
    //run_randomsearch(objective_name + "_random.csv"s,    objective, n_dims);
    run_benchmark<usdg::EI_AEI>(   objective_name + "_EI_AEI.csv"s,    objective, n_dims, logger.get());
    //run_benchmark<usdg::AEI_AEI>(  objective_name + "_AEI_AEI.csv"s,   objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Random>(objective_name + "_EI_Random.csv"s, objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Koyama>(objective_name + "_EI_Koyama.csv"s, objective, n_dims, logger.get());
  }

  {
    auto objective      = rosenbrock;
    size_t n_dims       = 20;
    auto objective_name = "rosenbrock20D"s;
    //run_randomsearch(objective_name + "_random.csv"s,    objective, n_dims);
    run_benchmark<usdg::EI_AEI>(   objective_name + "_EI_AEI.csv"s,    objective, n_dims, logger.get());
    //run_benchmark<usdg::AEI_AEI>(  objective_name + "_AEI_AEI.csv"s,   objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Random>(objective_name + "_EI_Random.csv"s, objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Koyama>(objective_name + "_EI_Koyama.csv"s, objective, n_dims, logger.get());
  }

  {
    auto objective      = ackley;
    size_t n_dims       = 10;
    auto objective_name = "ackley10D"s;
    //run_randomsearch(objective_name + "_random.csv"s,    objective, n_dims);
    run_benchmark<usdg::EI_AEI>(   objective_name + "_EI_AEI.csv"s,    objective, n_dims, logger.get());
    //run_benchmark<usdg::AEI_AEI>(  objective_name + "_AEI_AEI.csv"s,   objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Random>(objective_name + "_EI_Random.csv"s, objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Koyama>(objective_name + "_EI_Koyama.csv"s, objective, n_dims, logger.get());
  }

  {
    auto objective      = ackley;
    size_t n_dims       = 20;
    auto objective_name = "ackley20D"s;
    //run_randomsearch(objective_name + "_random.csv"s,    objective, n_dims);
    run_benchmark<usdg::EI_AEI>(   objective_name + "_EI_AEI.csv"s,    objective, n_dims, logger.get());
    //run_benchmark<usdg::AEI_AEI>(  objective_name + "_AEI_AEI.csv"s,   objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Random>(objective_name + "_EI_Random.csv"s, objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Koyama>(objective_name + "_EI_Koyama.csv"s, objective, n_dims, logger.get());
  }

  {
    auto objective      = hartmann;
    size_t n_dims       = 6;
    auto objective_name = "hartmann6D"s;
    //run_randomsearch(objective_name + "_random.csv"s,    objective, n_dims);
    run_benchmark<usdg::EI_AEI>(   objective_name + "_EI_AEI.csv"s,    objective, n_dims, logger.get());
    //run_benchmark<usdg::AEI_AEI>(  objective_name + "_AEI_AEI.csv"s,   objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Random>(objective_name + "_EI_Random.csv"s, objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Koyama>(objective_name + "_EI_Koyama.csv"s, objective, n_dims, logger.get());
  }

  {
    auto objective      = stablinskytang;
    size_t n_dims       = 10;
    auto objective_name = "stablinskytang10D"s;
    //run_randomsearch(objective_name + "_random.csv"s,    objective, n_dims);
    run_benchmark<usdg::EI_AEI>(   objective_name + "_EI_AEI.csv"s,    objective, n_dims, logger.get());
    //run_benchmark<usdg::AEI_AEI>(  objective_name + "_AEI_AEI.csv"s,   objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Random>(objective_name + "_EI_Random.csv"s, objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Koyama>(objective_name + "_EI_Koyama.csv"s, objective, n_dims, logger.get());
  }

  {
    auto objective      = stablinskytang;
    size_t n_dims       = 20;
    auto objective_name = "stablinskytang20D"s;
    //run_randomsearch(objective_name + "_random.csv"s,    objective, n_dims);
    run_benchmark<usdg::EI_AEI>(   objective_name + "_EI_AEI.csv"s,    objective, n_dims, logger.get());
    //run_benchmark<usdg::AEI_AEI>(  objective_name + "_AEI_AEI.csv"s,   objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Random>(objective_name + "_EI_Random.csv"s, objective, n_dims, logger.get());
    run_benchmark<usdg::EI_Koyama>(objective_name + "_EI_Koyama.csv"s, objective, n_dims, logger.get());
  }
}
