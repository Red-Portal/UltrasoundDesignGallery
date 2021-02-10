
#ifndef __US_GALLERY_ESS_HPP__
#define __US_GALLERY_ESS_HPP__

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>

#include <algorithm>
#include <random>
#include <stats.hpp>

namespace infer
{
  template <typename Rng, typename Func>
  inline blaze::DynamicVector<double>
  ess(Rng rng, Func f, double lb, double ub,
      size_t n_samples, size_t n_burn, size_t n_thin)
  {
    //auto samples = blaze::DynamicVector<double>(n_samples / n_thin);
  }
}

#endif
