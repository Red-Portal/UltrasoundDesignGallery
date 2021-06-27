
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

#include "lpndsf.hpp"
#include "pyramid.hpp"

namespace usdg
{
  LPNDSF::
  LPNDSF(size_t n_rows, size_t n_cols)
    : _diffusion0(),
      _diffusion1(), 
      _diffusion2(),
      _diffusion3(),
      _gaussian_pyramid_input(),
      _laplacian_pyramid_input(),
      _laplacian_pyramid_output()
  {
    auto [G_in, L_in, L_out]  = usdg::init_pyramid(n_rows, n_cols, 4);
    _gaussian_pyramid_input   = std::move(G_in);
    _laplacian_pyramid_input  = std::move(L_in);
    _laplacian_pyramid_output = std::move(L_out);

    _diffusion0.preallocate(static_cast<size_t>(_laplacian_pyramid_input[0].rows),
			    static_cast<size_t>(_laplacian_pyramid_input[0].cols));
    _diffusion1.preallocate(static_cast<size_t>(_laplacian_pyramid_input[1].rows),
			    static_cast<size_t>(_laplacian_pyramid_input[1].cols));
    _diffusion2.preallocate(static_cast<size_t>(_laplacian_pyramid_input[2].rows),
			    static_cast<size_t>(_laplacian_pyramid_input[2].cols));
    _diffusion3.preallocate(static_cast<size_t>(_laplacian_pyramid_input[3].rows),
			    static_cast<size_t>(_laplacian_pyramid_input[3].cols));
  }

  void
  LPNDSF::
  apply(cv::Mat const& image,
	cv::Mat& output,
	float t0, float sigma_g0, float sigma_r0,
	float t1, float sigma_g1, float sigma_r1,
	float t2, float sigma_g2, float sigma_r2,
	float t3, float sigma_g3, float sigma_r3,
	float alpha, float beta)
  {
    int n_padded_rows = _gaussian_pyramid_input[0].rows;
    int n_padded_cols = _gaussian_pyramid_input[0].cols;
    int n_row_pad     = n_padded_rows - image.rows;
    int n_col_pad     = n_padded_cols - image.cols;
    cv::copyMakeBorder(image,
		       _gaussian_pyramid_input[0],
		       0, n_row_pad,
		       0, n_col_pad,
		       cv::BORDER_CONSTANT,
		       cv::Scalar(0));
    usdg::analyze_pyramid(_gaussian_pyramid_input,
			  _laplacian_pyramid_input);

    float ctang0 = 0.1;
    float ctang1 = 0.1;
    float ctang2 = 0.1;
    float ctang3 = 0.1;

    float dt = 0.3f;
    size_t i = 0;
    _diffusion0.apply(_gaussian_pyramid_input[i],
		      _laplacian_pyramid_input[i],
		      _laplacian_pyramid_output[i],
		      dt, sigma_r0, sigma_g0, ctang0,
		      static_cast<size_t>(ceil(t0 / dt)));
    ++i;
    _diffusion1.apply(_gaussian_pyramid_input[i],
		      _laplacian_pyramid_input[i],
		      _laplacian_pyramid_output[i],
		      dt, sigma_r1, sigma_g1, ctang1,
		      static_cast<size_t>(ceil(t1 / dt)));
    ++i;
    _diffusion2.apply(_gaussian_pyramid_input[i],
		      _laplacian_pyramid_input[i],
		      _laplacian_pyramid_output[i],
		      dt, sigma_r2, sigma_g2, ctang2,
		      static_cast<size_t>(ceil(t2 / dt)));
    ++i;
    _diffusion3.apply(_gaussian_pyramid_input[i],
		      _laplacian_pyramid_input[i],
		      _laplacian_pyramid_output[i],
		      dt, sigma_r3, sigma_g3, ctang3,
		      static_cast<size_t>(ceil(t3 / dt)));

    auto& L3_lowpass_buf = _laplacian_pyramid_input[i];
    auto& L3_out         = _laplacian_pyramid_output[i];

    float eps   = 1e-5f;
    cv::log(L3_out + eps, L3_out);
    cv::GaussianBlur(L3_out, L3_lowpass_buf, cv::Size(5, 5), 1.0);
    auto L3_highpass = L3_out - L3_lowpass_buf;
    L3_lowpass_buf   = alpha*L3_lowpass_buf + beta*L3_highpass;
    cv::exp(L3_lowpass_buf, L3_out);

    usdg::synthesize_pyramid(_laplacian_pyramid_output);

    auto roi = cv::Rect(0, 0, image.cols, image.rows);
    _laplacian_pyramid_output[0](roi).copyTo(output);
  }
}
