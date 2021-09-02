
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

#ifndef _CUSTOM_IMAGE_PROCESSING_HPP_
#define _CUSTOM_IMAGE_PROCESSING_HPP_

#include <vector>
#include <string>

#include <opencv4/opencv2/core/utility.hpp>

#include "math/blaze.hpp"
#include "imaging/pipeline.hpp"

namespace usdg
{
  size_t const ee1_beta_idx   = 0;
  size_t const ee1_sigma_idx  = 1;
  size_t const ee2_beta_idx   = 2;
  size_t const ee2_sigma_idx  = 3;
  size_t const ncd1_s_idx     = 4;
  size_t const ncd1_alpha_idx = 5;
  size_t const ncd2_s_idx     = 6;
  size_t const ncd2_alpha_idx = 7;
  size_t const rpncd_k_idx    = 8;

  inline size_t 
  custom_ip_dimension()
  {
    return 9;
  }

  inline blaze::DynamicVector<double>
  custom_ip_scale()
  {
    return blaze::DynamicVector<double>(custom_ip_dimension(), 1.0);
  }

  inline float
  linear_interpolate(double x,
		     double x_min,
		     double x_max)
  {
    return static_cast<float>((x_max - x_min)*x + x_min);
  }

  inline float
  exp_interpolate(double x,
		  double x_min,
		  double x_max)
  { /* assumes x_min, x_max > 0 */
    double beta  = log(x_min);
    double alpha = log(x_max/x_min);
    return static_cast<float>(exp(alpha*x + beta));
  }

  inline std::vector<std::string>
  custom_ip_parameter_names()
  {
    auto param_names = std::vector<std::string>(custom_ip_dimension());
    param_names[ee1_beta_idx]   = "L1 enhance edge gain";
    param_names[ee1_sigma_idx]  = "L1 edge/detail threshold";
    param_names[ee2_beta_idx]   = "L0 enhance edge gain";
    param_names[ee2_sigma_idx]  = "L0 edge/detail threshold";
    param_names[ncd1_s_idx]     = "L2 NCD threshold";
    param_names[ncd1_alpha_idx] = "L2 NCD alpha";
    param_names[ncd2_s_idx]     = "L1 NCD threshold";
    param_names[ncd2_alpha_idx] = "L1 NCD alpha";
    param_names[rpncd_k_idx]    = "L0 RPNCD edge threshold";
    return param_names;
  }

  inline blaze::DynamicVector<float>
  custom_ip_transform_range(blaze::DynamicVector<double> const& param)
  {
    auto param_trans = blaze::DynamicVector<float>(param.size());
    param_trans[ee1_beta_idx]   = linear_interpolate(param[ee1_beta_idx],     1.0,  5.0);
    param_trans[ee1_sigma_idx]  = exp_interpolate(   param[ee1_sigma_idx],   1e-4, 1e-2);
    param_trans[ee2_beta_idx]   = linear_interpolate(param[ee2_beta_idx],     1.0,  5.0);
    param_trans[ee2_sigma_idx]  = exp_interpolate(   param[ee2_sigma_idx],   1e-4, 1e-2);
    param_trans[ncd1_s_idx]     = linear_interpolate(param[ncd1_s_idx],         1,  100);
    param_trans[ncd1_alpha_idx] = linear_interpolate(param[ncd1_alpha_idx],  3e-2,  0.1);
    param_trans[ncd2_s_idx]     = linear_interpolate(param[ncd2_s_idx],         1,  100);
    param_trans[ncd2_alpha_idx] = linear_interpolate(param[ncd2_alpha_idx],  3e-2,  0.1);
    param_trans[rpncd_k_idx]    = exp_interpolate(   param[rpncd_k_idx],     1e-4, 1e-2);
    return param_trans;
  }

  struct CustomImageProcessing
  {
    usdg::Pipeline _process;

    CustomImageProcessing(size_t n_rows,
			  size_t n_cols)
      : _process(n_rows, n_cols)
    { }

    inline void
    apply(cv::Mat const& input,
	  cv::Mat const& mask,
	  cv::Mat& output,
	  blaze::DynamicVector<double> const& param)
    {
      auto param_trans = custom_ip_transform_range(param);

      float ee_beta1   = param_trans[ee1_beta_idx];
      float ee_beta2   = param_trans[ee2_beta_idx];
      float ee_sigma1  = param_trans[ee1_sigma_idx];
      float ee_sigma2  = param_trans[ee2_sigma_idx];
      float ncd1_alpha = param_trans[ncd1_alpha_idx];
      float ncd1_s     = param_trans[ncd1_s_idx];
      float ncd2_alpha = param_trans[ncd2_alpha_idx];
      float ncd2_s     = param_trans[ncd2_s_idx];
      float rpncd_k    = param_trans[rpncd_k_idx];

      _process.apply(input, mask, output,
		     ee_beta1,
		     ee_sigma1,
		     ee_beta2,
		     ee_sigma2,
		     ncd1_s,
		     ncd1_alpha,
		     ncd2_s,
		     ncd2_alpha,
		     rpncd_k);
    }
  };
}

#endif
