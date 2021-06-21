
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
#include "custom_image_processing.hpp"
#include "metrics.hpp"

#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <csv.hpp>
#include <progressbar.hpp>
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
    auto alpha    = naive_linesearch(noisy_objective, lb, ub, 32);
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

    blaze::subvector(betas, 3, n_pseudo-3) = usdg::sample_beta(prng, alpha,
							       lb + eps,
							       ub - eps,
							       iter,
							       n_pseudo-3,
							       n_dims);
    betas[0] = lb;
    betas[1] = ub;
    betas[2] = 0.0;

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
  auto linescales = blaze::DynamicVector<double>(n_dims, 1.0);

  auto pb     = progressbar(static_cast<int>(n_reps)); 
  auto stream = std::ofstream(fname);
  auto writer = csv::make_csv_writer(stream);
  auto start  = usdg::clock::now();
//#pragma omp parallel for
  for (size_t i = 0; i < n_reps; ++i)
  {
//#pragma omp critical
    pb.update();
    auto prng = usdg::Random123(i);
    auto [hist_y, hist_t] = bayesian_optimization<Strategy>(
      prng, objective,  n_dims, n_init, n_iter, budget,
      n_pseudo, sigma, linescales, logger);

    auto& hist_y_local = hist_y;
//#pragma omp critical
    //writer << hist_y_local;
  }
  std::cout << std::endl;
  auto elapsed = usdg::compute_duration(start);
  logger->info("Finished running benchmark for {:.2f} seconds", elapsed.count());
}

void
phantom_objective(spdlog::logger* logger)
{
  using namespace std::string_literals;

  auto phantom = cv::imread("../data/phantom/field2_cyst_phantom.png");
  cv::cvtColor( phantom, phantom, cv::COLOR_RGB2GRAY);
  cv::normalize(phantom, phantom, 0, 1, cv::NORM_MINMAX, CV_32F);

  auto image_processing = usdg::CustomImageProcessing(static_cast<size_t>(phantom.rows),
						      static_cast<size_t>(phantom.cols));

  auto masks = std::vector<cv::Mat>(3);
  masks[0] = cv::imread("../data/phantom/class1_mask.png");
  masks[1] = cv::imread("../data/phantom/class2_mask.png");
  masks[2] = cv::imread("../data/phantom/class3_mask.png");
  for (size_t i = 0; i < 3; ++i)
  {
    cv::cvtColor(masks[i], masks[i], cv::COLOR_RGB2GRAY);
  }
  auto output = cv::Mat(phantom.rows, phantom.cols, CV_32F);

  auto objective = [&](blaze::DynamicVector<double> const& param){
    auto param_trans = usdg::custom_ip_transform_range(param);
    image_processing.apply(phantom, output, param_trans);
    return usdg::metrics::qindex(phantom, masks);
  };

  size_t n_dims = usdg::custom_ip_dimension();
  run_benchmark<usdg::EI_Koyama>("qindex_phantom.csv"s, objective, n_dims, logger);
}

int main()
{
  auto console  = spdlog::stdout_color_mt("console");
  spdlog::set_level(spdlog::level::info);
  auto logger  = spdlog::get("console");

  phantom_objective(logger.get());


  // using namespace std::string_literals;

  // auto image_processing = usdg::CustomImageProcessing();
  // auto objective = [&](blaze::DynamicVector<double> const& param){
  //   auto param_trans = custom_ip_transform_range(param);
  //   image_processing.apply(image, output, param_trans);
    
  //   return 
  // }

  // auto image  = cv::imread("");
  // cv::cvtColor( image, image, cv::COLOR_RGB2GRAY);
  // cv::normalize(image, image, 0, 1, cv::NORM_MINMAX, CV_32F);
  // auto output = cv::Mat(image.rows, image.cols, CV_32F);


  // [&](blaze::DynamicVector<double> const& param){
  //   auto param_trans = custom_ip_transform_range(param);
  //   image_processing.apply(image, output, param_trans);
    
  //   return 
  // }


  // auto output_rgba     = cv::Mat(_image_base.rows, _image_base.cols, CV_8UC4);
  // quantize(output_gray, output_quant);

  // auto console  = spdlog::stdout_color_mt("console");
  // spdlog::set_level(spdlog::level::info);
  // auto logger  = spdlog::get("console");

  // run_benchmark<usdg::EI_Koyama>(objective_name + "_EI_Koyama.csv"s, objective, n_dims, logger.get());
}
