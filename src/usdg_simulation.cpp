
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

#include <fstream>

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

#include <nlohmann/json.hpp>
#include <progressbar.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>

template <typename Func>
double
naive_linesearch(Func f, double lb, double ub, size_t budget)
{
  auto objective_lambda = [&f](std::vector<double> const& x,
			       std::vector<double>&) -> double
  {
    return f(x[0]);
  };

  auto objective_wrapped = std::function<
    double(std::vector<double> const&,
	   std::vector<double>&)>(objective_lambda);

  auto objective_invoke = +[](std::vector<double> const& x,
			      std::vector<double>& grad,
			      void* punned)
  {
    return (*reinterpret_cast<
	    std::function<
	    double(std::vector<double> const&,
		   std::vector<double>&)>*>(punned))(x, grad);
  };
  auto optimizer = nlopt::opt(nlopt::GN_DIRECT, 1);
  optimizer.set_max_objective(objective_invoke, &objective_wrapped);
  optimizer.set_xtol_rel(1e-3);
  optimizer.set_ftol_rel(1e-4);
  optimizer.set_maxeval(static_cast<int>(budget));
  optimizer.set_upper_bounds(ub);
  optimizer.set_lower_bounds(lb);

  auto x       = std::vector<double>({(ub + lb)/2});
  double y_buf = f(x[0]);
  optimizer.optimize(x, y_buf);
  return x[0];
}

template <typename Acq, typename Func, typename BudgetType>
std::tuple<std::vector<double>,
	   std::vector<blaze::DynamicVector<double>>,
	   std::vector<blaze::DynamicVector<double>>>
bayesian_optimization(usdg::Random123& prng,
		      Func objective,
		      size_t n_dims,
		      size_t n_init,
		      size_t n_iter,
		      BudgetType budget,
		      size_t n_pseudo,
		      blaze::DynamicVector<double> linescales,
		      spdlog::logger* logger)
{
  double eps      = 1e-2;
  auto profiler   = usdg::Profiler{};

  double y_opt    = std::numeric_limits<double>::lowest();
  auto x_opt      = blaze::DynamicVector<double>(n_dims);
  auto optimizer  = usdg::BayesianOptimization<Acq>(n_dims, n_pseudo);
  auto init_x     = optimizer.initial_queries(prng, n_init, logger);
  auto start      = usdg::clock::now();

  auto hist_x_opt       = std::vector<blaze::DynamicVector<double>>();
  auto hist_x_trans_opt = std::vector<blaze::DynamicVector<double>>();
  auto hist_y_opt       = std::vector<double>();
  for (auto& [x, xi] : init_x)
  {
    auto noisy_objective = [&, x=x, xi=xi](double alpha_in) {
      return objective(x + alpha_in*xi);
    };
    auto [lb, ub] = usdg::pbo_find_bounds(x, xi);
    auto alpha    = naive_linesearch(noisy_objective, lb, ub, 32);
    auto betas    = blaze::DynamicVector<double>(n_pseudo);
    auto y        = objective(x + xi*alpha);

    if (y > y_opt)
    {
      y_opt = y;
      x_opt = x + xi*alpha;
    }

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

  for (size_t iter = 1; iter < n_iter; ++iter)
  {
    auto [x, xi, _, __]  = optimizer.next_query(prng, iter, n_pseudo,
						budget, linescales,
						&profiler, logger);

    auto noisy_objective = [&, x=x, xi=xi](double alpha_in) {
      return objective(x + alpha_in*xi);
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

    if (y > y_opt)
    {
      y_opt = y;
      x_opt = x + xi*alpha;
    }
    optimizer.push_data(x, xi, betas, alpha);

    auto elapsed = usdg::compute_duration(start);
    hist_y_opt.push_back(y_opt);
    hist_x_opt.push_back(x_opt);
    hist_x_trans_opt.push_back(usdg::custom_ip_transform_range(x_opt));
    if (logger)
    {
      logger->info("iter = {:>4}, time = {:.2f}, y = {:g}, y opt = {:g}",
		   iter, elapsed.count(), y, y_opt);
    }
    std::cout << "x trans opt: " << hist_x_trans_opt.back() << std::endl;
  }
  return {std::move(hist_y_opt),
    std::move(hist_x_opt),
    std::move(hist_x_trans_opt)};
}

template <typename Strategy,
	  typename ObjFunc>
void
run_benchmark(std::string const& fname,
	      ObjFunc objective,
	      size_t n_dims,
	      size_t seed,
	      spdlog::logger* logger)
{
  logger->info("Running benchmark for {}", fname);

  size_t n_init   = 4;
  size_t n_iter   = 20;
  size_t budget   = 5000;
  size_t n_pseudo = 20;
  auto linescales = blaze::DynamicVector<double>(n_dims, 1.0);

  auto stream = std::ofstream(fname);
  auto data   = nlohmann::json();

  auto prng = usdg::Random123(seed);
  auto [hist_y, hist_x, hist_x_trans] = bayesian_optimization<Strategy>(
    prng, objective,  n_dims, n_init, n_iter, budget,
    n_pseudo, linescales, logger);

  auto data_rep = nlohmann::json();
  size_t data_len = hist_x.size();
  for (size_t t = 0; t < data_len; ++t)
  {
    auto data_iter = nlohmann::json();
    data_iter["y"]       = hist_y[t];
    data_iter["x"]       = std::vector<double>(hist_x[t].begin(),
					       hist_x[t].end());
    data_iter["x_trans"] = std::vector<double>(hist_x_trans[t].begin(),
					       hist_x_trans[t].end());
    data_rep.push_back(data_iter);
  }
  data.push_back(data_rep);
  stream << std::setw(2) << data << std::endl;;
  stream.close();
}

int main(int, char** argv)
{
  auto console  = spdlog::stdout_color_mt("console");
  spdlog::set_level(spdlog::level::info);
  auto logger   = spdlog::get("console");
  auto seed     = static_cast<size_t>(std::stoi(argv[1]));
  auto option   = static_cast<size_t>(std::stoi(argv[2]));

  /* load and prepare data */
  auto image     = cv::imread("../data/phantom/phantom_ATS549_2_1.pfm", cv::IMREAD_UNCHANGED);
  auto mask      = cv::imread("../data/phantom/phantom_ATS549_2_1.png", cv::IMREAD_GRAYSCALE);
  auto output    = cv::Mat(image.rows, image.cols, CV_32F);
  auto roi_masks = std::vector<cv::Mat>(6);
  roi_masks[0]   = cv::imread("../data/phantom/mask1.png", cv::IMREAD_GRAYSCALE);
  roi_masks[1]   = cv::imread("../data/phantom/mask2.png", cv::IMREAD_GRAYSCALE);
  roi_masks[2]   = cv::imread("../data/phantom/mask3.png", cv::IMREAD_GRAYSCALE);
  roi_masks[3]   = cv::imread("../data/phantom/mask4.png", cv::IMREAD_GRAYSCALE);
  roi_masks[4]   = cv::imread("../data/phantom/mask5.png", cv::IMREAD_GRAYSCALE);
  roi_masks[5]   = cv::imread("../data/phantom/mask6.png", cv::IMREAD_GRAYSCALE);

  /* image processing resources */
  auto image_processing = usdg::CustomImageProcessing(static_cast<size_t>(image.rows),
						      static_cast<size_t>(image.cols));

  {
    std::cout << "option: Q-index, seed: " << seed << std::endl;
    auto qindex = usdg::metrics::qindex(image, roi_masks);
    auto objective = [&](blaze::DynamicVector<double> const& param){
      image_processing.apply(image, mask, output, param);
      auto q    = usdg::metrics::qindex(output, roi_masks) / qindex;
      auto ssim = usdg::metrics::ssim(image, output);
      return q + 15*ssim[0];
    };
    run_benchmark<usdg::EI_Koyama>(std::string("q_index_optim.json"),
				   objective,
				   usdg::custom_ip_dimension(),
				   seed,
				   logger.get());
  }

  // {
  //   std::cout << "option: gCNR, seed: " << seed << std::endl;
  //   auto objective = [&](blaze::DynamicVector<double> const& param){
  //     image_processing.apply(image, output, param);
  //     return usdg::metrics::gcnr(output,
  // 				 masks[1],
  // 				 masks[2]);

  //   };
  //   run_benchmark<usdg::EI_Koyama>(std::string("gcnr_optim.json"),
  // 				   objective,
  // 				   usdg::custom_ip_dimension(),
  // 				   logger.get());
  // }

  // {
  //   std::cout << "option: SSNR, seed: " << seed << std::endl;
  //   auto objective = [&](blaze::DynamicVector<double> const& param){
  //     image_processing.apply(image, output, param);
  //     return usdg::metrics::ssnr(output, masks[2]);
  //   };
  //   run_benchmark<usdg::EI_Koyama>(std::string("ssnr_optim.json"),
  // 				   objective,
  // 				   usdg::custom_ip_dimension(),
  // 				   logger.get());
  // }
}
