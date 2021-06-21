
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
#include "imaging/lpndsf.hpp"

namespace usdg
{
  size_t const r0_idx = 0;
  size_t const r1_idx = 1;
  size_t const r2_idx = 2;

  size_t const k0_idx = 3;
  size_t const k1_idx = 4;
  size_t const k2_idx = 5;
  size_t const k3_idx = 6;

  size_t const t0_idx = 7;
  size_t const t1_idx = 8;
  size_t const t2_idx = 9;
  size_t const t3_idx = 10;

  size_t const alpha_idx = 11;
  size_t const beta_idx  = 12;

  inline size_t 
  custom_ip_dimension()
  {
    return 13;
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
    auto param_names    = std::vector<std::string>(custom_ip_dimension());
    param_names[r0_idx] = "r0 ";
    param_names[r1_idx] = "r1 ";
    param_names[r2_idx] = "r2 ";

    param_names[t0_idx] = "t0 ";
    param_names[t1_idx] = "t1 ";
    param_names[t2_idx] = "t2 ";
    param_names[t3_idx] = "t3 ";

    param_names[k0_idx] = "k0 ";
    param_names[k1_idx] = "k1 ";
    param_names[k2_idx] = "k2 ";
    param_names[k3_idx] = "k3 ";

    param_names[alpha_idx] = "alpha ";
    param_names[beta_idx]  = "beta ";
    return param_names;
  }

  inline blaze::DynamicVector<float>
  custom_ip_transform_range(blaze::DynamicVector<double> const& param)
  {
    auto param_trans    = blaze::DynamicVector<float>(param.size());
    param_trans[r0_idx] = linear_interpolate(param[r0_idx], 0.0, 0.5);
    param_trans[r1_idx] = linear_interpolate(param[r1_idx], 0.0, 0.5);
    param_trans[r2_idx] = linear_interpolate(param[r2_idx], 0.0, 0.5);

    param_trans[k0_idx] = exp_interpolate(param[k0_idx], 0.001, 1.0);
    param_trans[k1_idx] = exp_interpolate(param[k1_idx], 0.001, 1.0);
    param_trans[k2_idx] = exp_interpolate(param[k2_idx], 0.001, 1.0);
    param_trans[k3_idx] = exp_interpolate(param[k3_idx], 0.001, 1.0);

    param_trans[t0_idx] = linear_interpolate(param[t0_idx], 0.1, 10);
    param_trans[t1_idx] = linear_interpolate(param[t1_idx], 0.1, 10);
    param_trans[t2_idx] = linear_interpolate(param[t2_idx], 0.1, 10);
    param_trans[t3_idx] = linear_interpolate(param[t3_idx], 0.1, 10);

    param_trans[alpha_idx] = linear_interpolate(param[alpha_idx], 0.7, 1.0);
    param_trans[beta_idx]  = linear_interpolate(param[beta_idx],  1.0, 1.5);
    return param_trans;
  }

  struct CustomImageProcessing
  {
    usdg::LPNDSF _process;

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

      // std::cout << param.size() << std::endl;
      // std::cout << "r0: " << param_trans[r0_idx] << '\n'
      // 		<< "k0: " << param_trans[k0_idx] << '\n'
      // 		<< "t0: " << param_trans[t0_idx] << '\n'
      // 		<< "r1: " << param_trans[r1_idx] << '\n'
      // 		<< "k1: " << param_trans[k1_idx] << '\n'
      // 		<< "t1: " << param_trans[t1_idx] << '\n'
      // 		<< "r2: " << param_trans[r2_idx] << '\n'
      // 		<< "k2: " << param_trans[k2_idx] << '\n'
      // 		<< "t2: " << param_trans[t2_idx] << '\n'
      // 		<< "k3: " << param_trans[k3_idx] << '\n'
      // 		<< "t3: " << param_trans[t3_idx] << '\n'
      // 		<< "alpha: " << param_trans[alpha_idx] << '\n'
      // 		<< "beta : " << param_trans[beta_idx] << '\n'
      // 		<< std::endl;
      _process.apply(input, output,
		     param_trans[r0_idx],
		     param_trans[k0_idx],
		     param_trans[t0_idx],
		     param_trans[r1_idx],
		     param_trans[k1_idx],
		     param_trans[t1_idx],
		     param_trans[r2_idx],
		     param_trans[k2_idx],
		     param_trans[t2_idx],
		     param_trans[k3_idx],
		     param_trans[t3_idx],
		     param_trans[alpha_idx],
		     param_trans[beta_idx]);
    }
  };
}

#endif
