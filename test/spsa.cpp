
#include <catch2/catch.hpp>
#define BLAZE_USE_DEBUG_MODE 1

#include "../src/bo/spsa.hpp"
#include "../src/math/blaze.hpp"
#include "../src/math/prng.hpp"
#include "../src/math/mvnormal.hpp"

#include "utils.hpp"

TEST_CASE("minimize quadratic expression with spsa", "[spsa]")
{
  auto key  = GENERATE(range(0u, 8u));
  auto prng = usdg::Random123(key);

  size_t n_dims    = 5;
  auto normal_dist = std::normal_distribution<double>();

  auto A = generate_mvsamples(prng, n_dims, n_dims);
  A      = A*blaze::trans(A) + n_dims*blaze::IdentityMatrix<double>(n_dims);

  auto obj = [&](blaze::DynamicVector<double> const& x) -> double {
    return -blaze::dot(x, A*x) + 10*normal_dist(prng);
  };

  auto proj = [](blaze::DynamicVector<double> const& x)
    -> blaze::DynamicVector<double>
  {
    return x;
  };
  
  size_t n_iters  = 1000;
  double noise_sd = 10;
  double stepsize = 0.1;
  auto x_init     = usdg::rmvnormal(prng, n_dims);
  auto x_opt      = usdg::spsa_maximize(prng, obj, proj, noise_sd, stepsize,
					x_init, n_iters);
  REQUIRE( blaze::norm(x_opt - blaze::ZeroVector<double>(n_dims)) < 1e-1 );
}
