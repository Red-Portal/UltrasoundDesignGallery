
#include <catch2/catch.hpp>
#define BLAZE_USE_DEBUG_MODE 1

#include "../src/gp/sample_beta.hpp"
#include "../src/math/prng.hpp"

#include <iostream>

TEST_CASE("beta", "[linear algebra]")
{
  auto key  = GENERATE(range(0u, 8u));
  auto prng = usdg::Random123(key);
  double alpha = 0.5;
  double lb    = 0.0;
  double ub    = 1.0;
  size_t iter  = 50;
  size_t n_samples = 1024;
  size_t n_dims    = 4;

  std::cout << blaze::trans(usdg::sample_beta(prng, alpha, lb, ub, iter, n_samples, n_dims)) << std::endl;
}
