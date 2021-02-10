
#include <catch2/catch.hpp>

#include "../src/inference/imh.hpp"
#include "../src/inference/seed.hpp"

#include <iostream>

TEST_CASE("Sampling from unit Gaussian", "[imh]")
{
  auto seed = generate_seed(1);
  auto p    = [](double x){
    return stats::dnorm(x, 0.0, 1.0);
  };
  size_t n_samples = 4096;
  auto samples = infer::imh(seed, p, -3, 3, n_samples, 128, 2);

  double sample_mean_margin = 1.0 / sqrt(static_cast<double>(n_samples)) * 10;
  REQUIRE( blaze::mean(samples)   == Approx(0.0).margin(sample_mean_margin) );
  REQUIRE( blaze::stddev(samples) == Approx(1.0).margin(0.1) );
}
