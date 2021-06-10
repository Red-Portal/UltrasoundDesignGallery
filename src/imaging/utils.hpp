
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

#ifndef __US_GALLERY_IMAGING_UTILS_HPP__
#define __US_GALLERY_IMAGING_UTILS_HPP__

#define cuda_check(ans) { usdg::gpu_assert((ans), __FILE__, __LINE__); }

namespace usdg
{
  inline void
  gpu_assert(cudaError_t code,
	     const char *file,
	     int line,
	     bool abort=true)
  {
    if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n",
	      cudaGetErrorString(code), file, line);
      if (abort)
	exit(code);
    }
  }
}

#endif
