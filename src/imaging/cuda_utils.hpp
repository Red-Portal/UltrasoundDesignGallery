
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

#ifndef __US_GALLERY_CUDAUTILS_HPP__
#define __US_GALLERY_CUDAUTILS_HPP__

namespace usdg
{
  template <typename T>
  __device__ __forceinline__ void
  swap(T& a, T& b)
  {
    T c(a); a=b; b=c;
  }

  __device__ __forceinline__ void
  eigenbasis_2d(float A11, float A12, float A22,
		float& v1x, float& v1y,
		float& v2x, float& v2y,
		float& lambda1, float& lambda2)
		
  {
    float A_delta = (A11 - A22);
    float tmp     = sqrt(A_delta*A_delta + 4*A12*A12);
    v2x = 2*A12;
    v2y = A22 - A11 + tmp;

    float mag = sqrt(v2x*v2x + v2y*v2y);
    if (mag > 1e-7f)
    {
      v2x /= mag;
      v2y /= mag;
    }
    else
    {
      v2x = 1.0;
      v2y = 0.0;
    }

    v1x = -v2y;
    v1y = v2x;

    lambda1 = 0.5*(A11 + A22 - tmp);
    lambda2 = 0.5*(A11 + A22 + tmp);

    if (abs(lambda1) < abs(lambda2))
    {
      swap(lambda1, lambda2);
      swap(    v1x,     v2x);
      swap(    v1y,     v2y);
    }
  }

  __device__ __host__ __forceinline__ float2
  operator*(float a, float2 v)
  {
    float2 res;
    res.x = a*v.x;
    res.y = a*v.y;
    return res;
  }

  __device__ __host__ __forceinline__ float2
  operator*(float2 a, float2 b)
  {
    float2 res;
    res.x = (a.x*b.x) - (a.y*b.y);
    res.y = (a.x*b.y) + (a.y*b.x);
    return res;
  }

  __device__ __host__ __forceinline__ float2
  operator+(float2 a, float2 b)
  {
    float2 res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
  }

  __device__ __host__ __forceinline__ float2
  operator-(float2 a, float2 b)
  {
    float2 res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
  }

  __device__ __forceinline__ float
  tukey_biweight(float x, float sigma)
  {
    if (abs(x) < sigma)
    {
      float ratio = x / sigma;
      float coef  = 1.0 - ratio*ratio;
      return coef*coef;
    }
    else
    {
      return 0.0f;
    }
  }

  __device__ __forceinline__ int sign(float x)
  { 
    int t = x < 0 ? -1 : 0;
    return x > 0 ? 1 : t;
  }

  __device__ __forceinline__ float minmod(float x, float y)
  { 
    if(x*y > 0)
    {
      return sign(x)*min(abs(x), abs(y));
    }
    else
    {
      return 0.0;
    }
  }

  __device__ __forceinline__ float
  fetch_pixel(cv::cuda::PtrStepSzf img,
	      int i, int j,
	      cv::cuda::PtrStepSz<uchar> const mask,
	      float fallback)
  {
    if (mask(i,j) > 0)
      return img(i,j);
    else
      return fallback;
  }

  __device__ __forceinline__ float2
  fetch_pixel(cv::cuda::PtrStepSz<float2> img,
	      int i, int j,
	      cv::cuda::PtrStepSz<uchar> const mask,
	      float2 fallback)
  {
    if (mask(i,j) > 0)
      return img(i,j);
    else
      return fallback;
  }

  __device__ __forceinline__ void
  matrix_diffuse_impl(cv::cuda::PtrStepSzf       const img_src,
		      cv::cuda::PtrStepSz<uchar> const mask,
		      cv::cuda::PtrStepSzf       const D_xx,
		      cv::cuda::PtrStepSzf       const D_xy,
		      cv::cuda::PtrStepSzf       const D_yy,
		      int i, int j,
		      int xp, int xm, int yp, int ym,
		      int M, int N, float dt,
		      cv::cuda::PtrStepSzf img_dst)
  {
    float u5 = img_src(i, j);
    float u1 = fetch_pixel(img_src, xm, yp, mask, u5);
    float u2 = fetch_pixel(img_src,  i, yp, mask, u5);
    float u3 = fetch_pixel(img_src, xp, yp, mask, u5);
    float u4 = fetch_pixel(img_src, xm,  j, mask, u5);
    float u6 = fetch_pixel(img_src, xp,  j, mask, u5);
    float u7 = fetch_pixel(img_src, xm, ym, mask, u5);
    float u8 = fetch_pixel(img_src,  i, ym, mask, u5);
    float u9 = fetch_pixel(img_src, xp, ym, mask, u5);

    float a_c = D_xx(i, j);
    float b_c = D_xy(i, j);
    float c_c = D_yy(i, j);

    float a_xp = fetch_pixel(D_xx, xp, j, mask, a_c);
    float a_xm = fetch_pixel(D_xx, xm, j, mask, a_c);

    float b_xp = fetch_pixel(D_xy, xp,  j, mask, b_c);
    float b_xm = fetch_pixel(D_xy, xm,  j, mask, b_c);
    float b_yp = fetch_pixel(D_xy,  i, yp, mask, b_c);
    float b_ym = fetch_pixel(D_xy,  i, ym, mask, b_c);

    float c_yp = fetch_pixel(D_yy, i, yp, mask, c_c);
    float c_ym = fetch_pixel(D_yy, i, ym, mask, c_c);

    float c1 = (1.0/4)*(b_xm - b_yp);
    float c2 = (1.0/2)*(c_yp + c_c );
    float c3 = (1.0/4)*(b_xp + b_yp);
    float c4 = (1.0/2)*(a_xm + a_c );
    float c6 = (1.0/2)*(a_xp + a_c );
    float c7 = (1.0/4)*(b_xm + b_ym);
    float c8 = (1.0/2)*(c_ym + c_c );
    float c9 = (1.0/4)*(b_xp - b_ym);
    
    img_dst(i,j) = (u5 + dt*(c1*u1 + c2*u2 + c3*u3
			     + c4*u4 + c6*u6
			     + c7*u7 + c8*u8 + c9*u9)) /
      (1 + dt*(c1 + c2 + c3
	       + c4 + c6
	       + c7 + c8 + c9));
  }

}

#endif
