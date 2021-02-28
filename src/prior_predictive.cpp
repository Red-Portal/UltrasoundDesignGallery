
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

#include "gp/data.hpp"
#include "gp/gp_prior.hpp"
#include "gp/likelihood.hpp"
#include "inference/pm_ess.hpp"
#include "misc/mvnormal.hpp"
#include "misc/prng.hpp"
#include "misc/uniform.hpp"

#include <matplotlib-cpp/matplotlibcpp.h>
#include <spdlog/sinks/stdout_color_sinks.h>

std::tuple<usdg::LatentGaussianProcess<usdg::Matern52>,
	   blaze::DynamicMatrix<double>>
generate_gp_sample(usdg::Random123& prng,
		   usdg::Matern52 const& kernel,
		   size_t n_dims,
		   size_t n_init_points)
{
  auto dist       = std::normal_distribution<double>(0, 1);
  auto uniformgen = blaze::generate(
    n_dims, [&prng](size_t)->double { return usdg::runiform(prng); });

  auto init_points = blaze::DynamicMatrix<double>(n_init_points, n_dims); 
  for (size_t i = 0; i < n_init_points; ++i)
  {
    blaze::row(init_points, i) = blaze::trans(
      blaze::DynamicVector<double>(uniformgen));
  }
  auto K      = usdg::compute_gram_matrix(kernel, init_points);
  auto K_chol = usdg::cholesky_nothrow(K).value();
  auto z      = usdg::rmvnormal(prng, K.rows());
  auto y      = K_chol.L * z;
  auto alpha  = usdg::solve(K_chol, y);
  auto gp     = usdg::LatentGaussianProcess<usdg::Matern52>{
    std::move(K_chol), std::move(alpha), kernel};

  return {std::move(gp), std::move(init_points)};
}

usdg::Datapoint
generate_datapoint(usdg::Random123& prng,
		   blaze::DynamicMatrix<double> const& true_data,
		   usdg::LatentGaussianProcess<usdg::Matern52> const& true_gp,
		   size_t n_pseudo,
		   size_t n_dims,
		   double sigma)
{
  auto dist       = std::normal_distribution<double>(0, 1);
  auto uniformgen = blaze::generate(
    n_dims, [&prng](size_t)->double { return usdg::runiform(prng); });
  auto noise_dist = std::normal_distribution<double>(0, sigma);

  auto x   = blaze::DynamicVector<double>(uniformgen);
  auto xi  = usdg::rmvnormal(prng, n_dims);
  xi      /= blaze::max(blaze::abs(xi));

  auto [lb, ub]     = usdg::pgp_find_bounds(x, xi);
  auto coeff_dist   = std::uniform_real_distribution<double>(lb, ub);
  auto coefficients = blaze::DynamicVector<double>(n_pseudo + 1);
  for(size_t i = 0; i < n_pseudo + 1; ++i)
  {
    coefficients[i] = coeff_dist(prng);
  }

  auto observations = blaze::DynamicVector<double>(coefficients.size());
  for (size_t i = 0; i < observations.size(); ++i)
  {
    auto point       = blaze::evaluate(coefficients[i] *xi + x);
    auto [mean, var] = true_gp.predict(true_data, point);
    observations[i]  = mean + noise_dist(prng);
  }

  auto max_idx = blaze::argmax(observations);
  auto res  = usdg::Datapoint();
  res.betas = blaze::DynamicVector<double>(n_pseudo);
  res.xi    = std::move(xi);
  res.x     = std::move(x);
  res.alpha = 0.0;

  size_t beta_idx = 0;
  for(size_t i = 0; i < n_pseudo + 1; ++i)
  {
    if(i == max_idx)
    {
      res.alpha = coefficients[i];
    }
    else
    {
      res.betas[beta_idx] = coefficients[i];
      ++beta_idx;
    }
  }
  return res;
}

void
prior_predictive_check(usdg::Random123& prng,
		       size_t n_dims,
		       size_t n_pseudo,
		       size_t n_data,
		       usdg::MvNormal<usdg::DiagonalChol> const& prior_dist,
		       spdlog::logger* logger)
{
  auto hypers  = prior_dist.sample(prng);
  double sigma = exp(hypers[0]);
  auto true_kernel = usdg::Matern52{
    exp(hypers[1]), blaze::exp(blaze::subvector(hypers, 2, n_dims))};

  size_t n_init_points      = 512;
  auto [true_gp, true_data] = generate_gp_sample(prng, true_kernel, n_dims, n_init_points);

  if(logger)
  {
    logger->info("generating data points...");
  }
  auto data = usdg::Dataset(n_dims, n_pseudo);
  for (size_t i = 0; i < n_data; ++i)
  {
    data.push_back(
      generate_datapoint(prng, true_data, true_gp, n_pseudo, n_dims, sigma));
  }
  if(logger)
  {
    logger->info("generated data points");
  }

  double sigma_buf = exp(hypers[0]);
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

  auto data_mat  = data.data_matrix();
  auto make_gram = [&](blaze::DynamicVector<double> const& theta_in)
    ->blaze::DynamicMatrix<double>
    {
      sigma_buf   = exp(theta_in[0]);
      auto kernel = usdg::Matern52{
	exp(theta_in[1]), blaze::exp(blaze::subvector(theta_in, 2, n_dims))};
      return usdg::compute_gram_matrix(kernel, data_mat);
    };

  size_t n_samples = 64;
  size_t n_burn    = 64;

  auto [theta_samples, f_samples, K_samples] = usdg::pm_ess(
    prng,
    loglike,
    grad_hess,
    make_gram,
    prior_dist.mean,
    prior_dist,
    data_mat.rows(),
    n_samples,
    n_burn,
    logger);

  {
    std::vector<std::vector<double>> x_plt, y_plt, z_plt;
    for (size_t i = 0; i < 30; ++i) {
      std::vector<double> x_row, y_row, z_row;
      for (size_t j = 0; j < 30; ++j) {
	double xval = i*(1.0/30);
	double yval = j*(1.0/30);
	x_row.push_back(xval);
	y_row.push_back(yval);
	auto [mean, var] = true_gp.predict(true_data, blaze::DynamicVector<double>({xval, yval}));
	z_row.push_back(mean);
      }
      x_plt.push_back(x_row);
      y_plt.push_back(y_row);
      z_plt.push_back(z_row);
    }
    matplotlibcpp::plot_surface(x_plt,y_plt,z_plt);
    matplotlibcpp::show();
  }

  // {
  //   std::vector<std::vector<double>> x_plt, y_plt, z_plt;
  //   for (double i = 0; i <= 1.0;  i += 0.03) {
  //     std::vector<double> x_row, y_row, z_row;
  //     for (double j = 0; j <= 1.0; j += 0.03) {
  // 	x_row.push_back(i);
  // 	y_row.push_back(j);
  // 	std::cout << i << " " << j << std::endl;

  // 	double res = 0.0;
  // 	for (size_t k = 0; k < n_samples; ++k)
  // 	{
  // 	  auto theta  = blaze::column(theta_samples, k);
  // 	  auto kernel = usdg::Matern52{
  // 	    exp(theta[1]), blaze::exp(blaze::subvector(theta, 2, n_dims))};

  // 	  auto& K_i   = K_samples[k];
  // 	  auto alpha  = usdg::solve(K_i, blaze::column(f_samples, k));
  // 	  auto gp     = usdg::LatentGaussianProcess<usdg::Matern52>{K_i, alpha, kernel};
  // 	  auto [mean, var] = true_gp.predict(true_data, blaze::DynamicVector<double>({i, j}));
  // 	  res += mean;
  // 	}
  // 	z_row.push_back(res / static_cast<double>(n_samples));
  //     }
  //     x_plt.push_back(x_row);
  //     y_plt.push_back(y_row);
  //     z_plt.push_back(z_row);
  //   }
  //   matplotlibcpp::plot_surface(x_plt,y_plt,z_plt);
  //   matplotlibcpp::show();
  // }

  std::cout << theta_samples << std::endl;
}

int main()
{
  auto key        = 0u;
  auto prng       = usdg::Random123(key);
  size_t n_dims   = 2;
  size_t n_pseudo = 5;
  size_t n_data   = 50;

  auto prior_mean = blaze::DynamicVector<double>(n_dims+2, 0.0);
  prior_mean[0]   = -2.0;
  auto prior_var  = blaze::DynamicVector<double>(n_dims+2, 1.0);
  prior_var[0]    = 1.0; 

  auto prior_chol = usdg::cholesky_nothrow(prior_var).value();
  auto prior_dist = usdg::MvNormal<usdg::DiagonalChol>{prior_mean, prior_chol};

  auto console  = spdlog::stdout_color_mt("console");
  spdlog::set_level(spdlog::level::info);
  auto logger  = spdlog::get("console");

  prior_predictive_check(prng, n_dims, n_pseudo, n_data, prior_dist, logger.get());
}
