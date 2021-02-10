
#ifndef __US_GALLERY_GP_PRIOR_HPP__
#define __US_GALLERY_GP_PRIOR_HPP__

#include <blaze/math/DynamicVector.h>

namespace gp
{
  struct Hyperparams
  {
    double logstdlike;
    double logstdkernel;
    double logstdnoise;
    blaze::DynamicVector<double> ardscales;
  };
}

#endif
