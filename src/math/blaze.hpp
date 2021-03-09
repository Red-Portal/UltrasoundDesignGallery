
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

#ifndef __US_GALLERY_BLAZE_HPP__
#define __US_GALLERY_BLAZE_HPP__

#define BLAZE_BLAS_MODE 1
#define BLAZE_BLAS_IS_64BIT 1
#define BLAZE_BLAS_IS_PARALLEL 1
#define BLAZE_USE_BLAS_MATRIX_VECTOR_MULTIPLICATION 1
#define BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION 1
#define BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 0
#define BLAZE_USE_VECTORIZATION 1
#define BLAZE_DEFAULT_STORAGE_ORDER blaze::columnMajor

//#define BLAZE_USE_DEBUG_MODE 1

#include <blaze/Blaze.h>

#endif
