
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
  size_t const t_idx = 0;
  size_t const a_idx = 1;
  size_t const b_idx = 2;
  size_t const g_idx = 3;
  size_t const c_idx = 4;

  size_t const theta_idx = 5;
  size_t const alpha_idx = 6;
  size_t const beta_idx  = 7;

  inline size_t 
  custom_ip_dimension()
  {
    return 8;
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
    param_names[t_idx]     = "diffusion time";
    param_names[a_idx]     = "tissue prob a";
    param_names[b_idx]     = "tissue prob b";
    param_names[g_idx]     = "gradient threshold";
    param_names[c_idx]     = "ctang";
    param_names[theta_idx] = "theta";
    param_names[alpha_idx] = "alpha";
    param_names[beta_idx]  = "beta";
    return param_names;
  }

  inline blaze::DynamicVector<float>
  custom_ip_transform_range(blaze::DynamicVector<double> const& param)
  {
    auto param_trans   = blaze::DynamicVector<float>(param.size());
    param_trans[t_idx] = linear_interpolate(param[t_idx],      0.1,   20);
    param_trans[a_idx] = exp_interpolate(param[a_idx],         0.1, 10.0);
    param_trans[b_idx] = exp_interpolate(param[b_idx],         0.1, 10.0);
    param_trans[g_idx] = exp_interpolate(   param[g_idx],     0.01,  1.0);
    param_trans[c_idx] = exp_interpolate(   param[c_idx],    0.001,    5);

    param_trans[theta_idx] = linear_interpolate(param[theta_idx], 0.1,  16);
    param_trans[alpha_idx] = linear_interpolate(param[alpha_idx], 0.5, 1.0);
    param_trans[beta_idx]  = linear_interpolate(param[beta_idx],  1.0, 2.0);
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
	  cv::Mat& output,
	  blaze::DynamicVector<double> const& param)
    {
      auto param_trans = custom_ip_transform_range(param);
      _process.apply(input, output,
		     param_trans[t_idx],
		     param_trans[a_idx],
		     param_trans[b_idx],
		     param_trans[g_idx],
		     param_trans[c_idx],
		     param_trans[theta_idx],
		     param_trans[alpha_idx],
		     param_trans[beta_idx]);
    }
  };
}

#endif
