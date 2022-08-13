
/*
 * Copyright (C) 2021-2022 Kyurae Kim
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

#include "catch.hpp"
#define BLAZE_USE_DEBUG_MODE 1

#include "../src/bo/find_bounds.hpp"
#include "../src/gp/gp_prior.hpp"
#include "../src/math/cholesky.hpp"
#include "../src/math/mvnormal.hpp"
#include "../src/math/prng.hpp"
#include "../src/system/profile.hpp"

#include "finitediff.hpp"
#include "utils.hpp"

TEST_CASE("gaussian process mean gradient", "[gp]")
{
  auto key        = GENERATE(range(0u, 8u));
  auto prng       = usdg::Random123(key);
  size_t n_dims   = 8;
  size_t n_points = 32;

  auto data_mat  = generate_mvsamples(prng, n_dims, n_points);
  auto norm_dist = std::normal_distribution<double>(0, 1);
  auto sigma     = exp(norm_dist(prng));
  auto scale     = exp(norm_dist(prng) + 1);
  auto kernel    = usdg::SquaredExpIso(sigma, scale);
  auto gram      = usdg::compute_gram_matrix(kernel, data_mat);
  auto z         = usdg::rmvnormal(prng, n_points);
  auto dx        = usdg::rmvnormal(prng, n_dims);
  auto chol      = usdg::cholesky_nothrow(gram).value();

  auto gp         = usdg::GP<decltype(kernel)>{
    std::move(chol), z, kernel};
  auto grad_truth = finitediff_gradient(
    [&data_mat, &gp](blaze::DynamicVector<double> const& x)
    {
      auto [mean, var] = gp.predict(data_mat, x);
      return mean;
    }, dx);
  auto [mean, grad]    = usdg::gradient_mean(gp, data_mat, dx);
  auto [mean_truth, _] = gp.predict(data_mat, dx);

  REQUIRE( blaze::norm(grad_truth - grad) < 1e-4 );
  REQUIRE( mean == usdg::Approx(mean_truth) );
}

TEST_CASE("gaussian process prediction mean and variance gradient", "[gp]")
{
  auto key        = GENERATE(range(0u, 8u));
  auto prng       = usdg::Random123(key);
  size_t n_dims   = 8;
  size_t n_points = 32;

  auto data_mat  = generate_mvsamples(prng, n_dims, n_points);
  auto norm_dist = std::normal_distribution<double>(0, 1);
  auto sigma     = exp(norm_dist(prng));
  auto scale     = exp(norm_dist(prng) + 1);
  auto kernel    = usdg::SquaredExpIso(sigma, scale);
  auto gram      = usdg::compute_gram_matrix(kernel, data_mat);
  auto z         = usdg::rmvnormal(prng, n_points);
  auto dx        = usdg::rmvnormal(prng, n_dims);
  auto chol      = usdg::cholesky_nothrow(gram).value();

  auto gp         = usdg::GP<decltype(kernel)>{
    std::move(chol), z, kernel};


  auto mean_grad_truth = finitediff_gradient(
    [&data_mat, &gp](blaze::DynamicVector<double> const& x)
    {
      auto [mean, var] = gp.predict(data_mat, x);
      return mean;
    }, dx);
  auto var_grad_truth = finitediff_gradient(
    [&data_mat, &gp](blaze::DynamicVector<double> const& x)
    {
      auto [mean, var] = gp.predict(data_mat, x);
      return var;
    }, dx);
  auto [mean_truth, var_truth]          = gp.predict(data_mat, dx);
  auto [mean, var, mean_grad, var_grad] = usdg::gradient_mean_var(gp, data_mat, dx);

  REQUIRE( mean == usdg::Approx(mean_truth) );
  REQUIRE( var  == usdg::Approx(var_truth) );
  REQUIRE( blaze::norm(mean_grad_truth - mean_grad) < 1e-4 );
  REQUIRE( blaze::norm(var_grad_truth  - var_grad)  < 1e-4 );
}

TEST_CASE("gaussian process batch prediction", "[gp]")
{
  auto key        = GENERATE(range(0u, 8u));
  auto prng       = usdg::Random123(key);
  size_t n_dims   = 8;
  size_t n_points = 32;

  auto data_mat  = generate_mvsamples(prng, n_dims, n_points);
  auto norm_dist = std::normal_distribution<double>(0, 1);
  auto sigma     = exp(norm_dist(prng));
  auto scale     = exp(norm_dist(prng) + 1);
  auto kernel    = usdg::SquaredExpIso(sigma, scale);
  auto gram      = usdg::compute_gram_matrix(kernel, data_mat);
  auto z         = usdg::rmvnormal(prng, n_points);
  auto dx        = usdg::rmvnormal(prng, n_dims);
  auto chol      = usdg::cholesky_nothrow(gram).value();
  auto gp        = usdg::GP<decltype(kernel)>{std::move(chol), z, kernel};

  size_t n_x       = 128;
  auto means_truth = blaze::DynamicVector<double>(n_x);
  auto vars_truth  = blaze::DynamicVector<double>(n_x);
  auto X           = generate_mvsamples(prng, n_dims, n_x);

  for (size_t i = 0; i < X.columns(); ++i)
  {
    auto [mean, var] = gp.predict(data_mat, blaze::column(X, i));
    means_truth[i] = mean;
    vars_truth[i]  = var;
  }
  auto [means, vars] = gp.predict(data_mat, X);
  for (size_t i = 0; i < X.columns(); ++i)
  {
    REQUIRE( means[i] == usdg::Approx(means_truth[i]) );
    REQUIRE( vars[i]  == usdg::Approx(vars_truth[i]) );
  }
}

TEST_CASE("gaussian process prediction gradient regression1", "[gp]")
{
  auto data_mat  = blaze::DynamicMatrix<double>{
    {0.835248,  0.841446,   0.843102,   0.84353,  0.838296,  0.649214, 0.651899,  0.640166,   0.682612, 0.651628, 0.534399,  0.510567,  0.52977,   0.50226, 0.517407,   0.41023,  0.493844,  0.466412,   0.54297,  0.467464,  0.524246,   0.47388,  0.454902,  0.517634,  0.470901,   0.29128,  0.290777,   0.279564,  0.280481,    0.278918,      0.73195,   0.758865,  0.783234,  0.784758,  0.76013,  0.957462,    0.953157,  0.906225,  0.975952,  0.983839 },
    { 0.14754, 0.0550352,   0.030312, 0.0239211,  0.102051, 0.0460455,0.0518333, 0.0265448,   0.118028,0.0512483, 0.545141,  0.727511, 0.580567,  0.791076, 0.675165,  0.526773,  0.531119,  0.529693,  0.533672,  0.529748,  0.726209,  0.851377,   0.89854,  0.742642,   0.85878,  0.931842,  0.934073,   0.983793,  0.979726,    0.986655,     0.956598,   0.932038,  0.909801,   0.90841, 0.930884,  0.799782,    0.813832,  0.967005,  0.739437,  0.713696 },
    {.0895927, 0.0739904,  0.0698205, 0.0687426, 0.0819203,  0.385953, 0.378285,  0.411786,   0.290594,  0.37906, 0.607362,  0.824108, 0.649466,  0.899654, 0.761895,  0.225104,  0.330109,   0.29566,  0.391804,  0.296981,  0.209557,    0.1455,  0.121363,  0.201147,  0.141711,  0.462324,  0.465593,   0.538449,  0.532488,    0.542642,    0.0631738,   0.106315,  0.145375,  0.147819, 0.108342,  0.184294,    0.181981,  0.156767,  0.194228,  0.198465 },
    {       0, 0.0550031,  0.0697035, 0.0735035, 0.0270476,   0.63901, 0.632783,  0.659993,   0.561558, 0.633412, 0.878083,  0.936192, 0.889371,  0.956446, 0.919513,  0.805349,  0.886806,  0.860082,  0.934665,  0.861107,  0.416071,  0.481217,  0.505764,  0.424624,   0.48507, 0.0726875, 0.0744787,   0.114399,  0.111133,    0.116697,   0.00464252,  0.0266895, 0.0466505, 0.0478992, .0277253,   0.32621,    0.330041,  0.371801,  0.309758,   0.30274 },
    { 0.91694,  0.895249,   0.889452,  0.887953,  0.906274,   0.45158, 0.452931,   0.44703,   0.468376, 0.452794, 0.582104,  0.281526, 0.523716,   0.17676, 0.367801,     0.461,  0.722734,  0.636866,  0.876514,  0.640159, 0.0920388, 0.0768545, 0.0711331, 0.0900452, 0.0759564, 0.0495186, 0.0532097,    0.13547,   0.12874,    0.140204, -3.46945e-18,   0.076477,  0.145718,   0.15005,  0.08007,  0.633369,    0.633851,  0.639099,  0.631302,   0.63042 },
    {0.517417,  0.425278,   0.400653,  0.394287,  0.472108,  0.441032,  0.44913,  0.413746,    0.54175, 0.448312, 0.558648,   0.66777, 0.579845,  0.705804, 0.636449,  0.136472, 0.0924512,  0.106893, 0.0665871,   0.10634, 0.0101745,  0.155569,  0.210353, 0.0292632,  0.164169,  0.114655,  0.110024, 0.00681622, 0.0152599, 0.000876107,     0.813833,     0.9003,  0.978587,  0.983484, 0.904362,  0.983316,    0.980862,   0.95411,  0.993856,  0.998352 },
    {0.177302,  0.176741,   0.176591,  0.176552,  0.177026,  0.958023, 0.959092,   0.95442,   0.971323, 0.958984,  0.80518,  0.811562,  0.80642,  0.813786,  0.80973,  0.549291,  0.537253,  0.541203,  0.530181,  0.541051,   0.99203,  0.944078,  0.926009,  0.985734,  0.941241,  0.673842,  0.672806,   0.649707,  0.651596,    0.648377,     0.207595,   0.165859,  0.128072,  0.125708, 0.163898, 0.0760211,   0.0707776, 0.0136114, 0.0985427,   0.10815 },
    {0.226812,  0.187043,   0.176415,  0.173667,  0.207256,  0.522944, 0.525947,  0.512827,   0.560291, 0.525644, 0.992905,  0.927238, 0.980149,   0.90435, 0.946086,  0.271682,  0.231417,  0.244627,   0.20776,   0.24412,  0.709366,  0.506839,  0.430526,  0.682777,   0.49486,  0.104461,  0.108333,   0.194629,  0.187569,    0.199596,     0.222675,   0.211732,  0.201825,  0.201205, 0.211218,  0.506967,    0.503767,  0.468872,  0.520714,  0.526579 },
    {0.273551,  0.274925,   0.275292,  0.275387,  0.274226,  0.564507, 0.565091,   0.56254,   0.571767, 0.565032, 0.532168,  0.460624,  0.51827,  0.435688, 0.481159,  0.815546,  0.721747,   0.75252,  0.666636,   0.75134,  0.702128,  0.655268,  0.637611,  0.695976,  0.652496,  0.185319,  0.184228,   0.159915,  0.161904,    0.158515,     0.208758,   0.179694,   0.15338,  0.151734, 0.178329,  0.130995,    0.132046,  0.143504,  0.126481,  0.124555 },
    {0.569178,   0.64974,   0.671271,  0.676837,  0.608794,  0.120122,  0.11063,  0.152104, 0.00206796, 0.111589, 0.999209,  0.937241, 0.987171,  0.915643, 0.955028,  0.562841,  0.374935,  0.436582,  0.264533,  0.434218,  0.874188,     0.768,  0.727989,  0.860247,  0.761719,         1,  0.996724,   0.923705,  0.929679,    0.919503,     0.890152,   0.908608,  0.925318,  0.926363, 0.909475,  0.638931,    0.633067,   0.56913,   0.66412,  0.674864 }};

  auto alpha = {
    0.589814, -0.147447, -0.147527, -0.147548, -0.147291, 0.589042, -0.147292,
    -0.147366, -0.147091, -0.147294, 0.582914, -0.14503, -0.148011, -0.143775,
    -0.146098, 0.581062, -0.145176, -0.146448, -0.143039, -0.146399, 0.584068,
    -0.145733, -0.144669, -0.148098, -0.145567, 0.587395, -0.147635, -0.146578,
    -0.146666, -0.146516, 0.58616, -0.146998, -0.146131, -0.146078, -0.146953,
    0.589039, -0.147224, -0.146301, -0.147679, -0.147836};

  auto dx        = blaze::DynamicVector<double>({
      0.524246, 0.726209,0.209557, 0.416071, 0.0920388, 0.0101745, 0.99203,
      0.709366, 0.702128, 0.874188});
  dx += blaze::DynamicVector<double>(dx.size(), 0.001);

  auto sigma     = 0.626403;
  auto scale     = 2.69178;
  auto kernel    = usdg::SquaredExpIso{sigma, scale};
  auto gram      = usdg::compute_gram_matrix(kernel, data_mat);
  auto chol      = usdg::cholesky_nothrow(gram).value();
  auto gp        = usdg::GP<decltype(kernel)>{
    std::move(chol), std::move(alpha), std::move(kernel)};
  auto grad_truth = finitediff_gradient(
    [&data_mat, &gp](blaze::DynamicVector<double> const& x)
    {
      auto [mean, var] = gp.predict(data_mat, x);
      return mean;
    }, dx);
  auto [mean, grad]    = usdg::gradient_mean(gp, data_mat, dx);
  auto [mean_truth, _] = gp.predict(data_mat, dx);

  REQUIRE( blaze::norm(grad_truth - grad) < 1e-4 );
  REQUIRE( mean == usdg::Approx(mean_truth) );
}

