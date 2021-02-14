
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

#include <catch2/catch.hpp>

#include <limits>

#include "../src/gp/kernel.hpp"
#include "../src/gp/data.hpp"
#include "../src/gp/gp_prior.hpp"

double const catch_eps = 1e-8;

TEST_CASE("Matern 5/2 kernel value", "[kernel]")
{
  /* 
   * Compared against Julia KernelFunctions.jl result

   using KernelFunctions
   ℓ = [0.3468645418042131, 0.4089279965964634]; σ² = 0.6
   k = KernelFunctions.Matern52Kernel()
   t = KernelFunctions.ARDTransform(1 ./ℓ)
   k = σ²*KernelFunctions.transform(k, t)
   x = [1.124618098544101, -1.8477787735615157]; y = [1.0597907259031794, 0.20131396456561368]
   k(x, y)

   */
  auto linescales = blaze::DynamicVector<double>({0.3468645418042131, 0.4089279965964634});
  auto sigma      = sqrt(0.6);
  auto kernel     = usvg::Matern52{sigma, linescales};

  auto x = blaze::DynamicVector<double>({1.124618098544101, -1.8477787735615157});
  auto y = blaze::DynamicVector<double>({1.0597907259031794, 0.20131396456561368});

  REQUIRE(kernel(x, y) == Approx(0.0004385141317002246));
}

TEST_CASE("Gram matrix computation", "[kernel]")
{
  /* 
   * Compared against Julia KernelFunctions.jl result

   using KernelFunctions
   ℓ = [0.3468645418042131, 0.4089279965964634]; σ² = 0.6
   k = KernelFunctions.Matern52Kernel()
   t = KernelFunctions.ARDTransform(1 ./ℓ)
   k = σ²*KernelFunctions.transform(k, t)
   x = [1.124618098544101, -1.8477787735615157]; y = [1.0597907259031794, 0.20131396456561368]

   data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
   kernelmatrix(k, data, obsdim=1)
   */

  auto linescales = blaze::DynamicVector<double>(
    {0.3468645418042131, 0.4089279965964634});
  auto sigma      = sqrt(0.6);
  auto kernel     = usvg::Matern52{sigma, linescales};
  auto datamatrix = blaze::DynamicMatrix<double>(
    {{1.0, 3.0, 5.0},
     {2.0, 4.0, 6.0}});
  auto truth = blaze::DynamicMatrix<double>(
    {{0.6,         3.08681e-6, 5.15601e-13},
     {3.08681e-6,  0.6,        3.08681e-6},
     {5.15601e-13, 3.08681e-6, 0.6}});

  auto gram = usvg::compute_gram_matrix(kernel, datamatrix);
  REQUIRE( blaze::norm(gram - truth) < catch_eps );
}


TEST_CASE("Data matrix construction", "[dataset]")
{
  auto x1     = blaze::DynamicVector<double>({0.3, 0.4});
  auto xi1    = blaze::DynamicVector<double>({1.0, 2.0});
  auto alpha1 = 0.2;
  auto betas1 = blaze::DynamicVector<double>({0.1, 0.3});
  /*  
   *  [ 0.5 0.8;  alpha
   *    0.4 0.6;  beta1
   *    0.6 1.0 ] beta2
   */

  auto x2     = blaze::DynamicVector<double>({0.3, 0.4});
  auto xi2    = blaze::DynamicVector<double>({-1.0, -2.0});
  auto alpha2 = 0.1;
  auto betas2 = blaze::DynamicVector<double>({0.2, 0.4});
  /*  
   *  [ 0.2  0.2;  alpha
   *    0.1  0.0;  beta1
   *   -0.1 -0.4 ] beta2
   */

  size_t n_dims   = 2; 
  size_t n_pseudo = 2;
  size_t n_data   = 2;
  auto dataset     = usvg::Dataset(n_dims, n_pseudo);
  auto data_matrix = blaze::DynamicMatrix<double>();
  REQUIRE_NOTHROW( dataset.push_back(usvg::Datapoint{alpha1, betas1, xi1, x1}) );
  REQUIRE_NOTHROW( dataset.push_back(usvg::Datapoint{alpha2, betas2, xi2, x2}) );
  REQUIRE_NOTHROW( data_matrix = dataset.data_matrix() );

  REQUIRE( data_matrix.rows()    == n_dims );
  REQUIRE( data_matrix.columns() == n_data*(1 + n_pseudo) );

  auto truth = blaze::DynamicMatrix<double>({{0.5, 0.8},
					     {0.4, 0.6},
					     {0.6, 1.0},

					     {0.2,  0.2},
					     {0.1,  0.0},
					     {-0.1, -0.4}});

  REQUIRE( blaze::norm(blaze::trans(data_matrix) - truth) < catch_eps );
}

TEST_CASE("Latent Gaussian process prediction", "[dataset]")
{
  auto data = blaze::DynamicMatrix<double>(
    {{1, 3,  1},
     {2, -1, 1},
     {-1, 2, 1}});
  auto linescales = blaze::DynamicVector<double>({1.0, 1.0, 1.0});
  auto sigma      = sqrt(0.6);
  auto kernel     = usvg::Matern52{sigma, linescales};

  auto K = usvg::compute_gram_matrix(kernel, data);
  auto W = blaze::DynamicMatrix<double>(
    {{  1, 0.1, 0.1},
     {0.1,   2, 0.1},
     {0.1, 0.1,   3}});

  auto f     = blaze::DynamicVector<double>(
    {0.9040983839157295,
     -0.29874050736604413,
     -1.2570687585683156});

  auto WK     = W*K;
  auto K_chol = usvg::Cholesky<usvg::DenseChol>();
  REQUIRE_NOTHROW( K_chol = usvg::cholesky_nothrow(K).value() );
  auto alpha  = usvg::solve(K_chol, f);
  auto IpWK   = usvg::lu(blaze::IdentityMatrix<double>(WK.rows()) + WK);
  
  auto gp = usvg::LatentGaussianProcess<usvg::Matern52>{
    K_chol, alpha, data, WK, IpWK, kernel };

  auto x = blaze::DynamicVector<double>(
    {1.624602457143822,
     -0.3130882688487052,
     -0.8858236880999151});
  auto [mean, var] = gp.predict(x);
  
  auto k_star   = blaze::DynamicVector<double>(3);
  for (size_t i = 0; i < 3; ++i)
  {
    k_star[i] = kernel(blaze::column(data, i), x);
  }
  auto mean_truth = blaze::dot(k_star, alpha);
  REQUIRE( mean == Approx(mean_truth) );

  blaze::invert(W);
  auto var_truth = kernel(x, x) - blaze::dot(k_star, blaze::solve(K + W, k_star));
  REQUIRE( var == Approx(var_truth).margin(1e-2) );
}
