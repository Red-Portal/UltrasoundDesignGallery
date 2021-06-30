
import Images
import ImageView
import FileIO
import Plots, StatsPlots
import MosaicViews
import ImageCore
import ProgressMeter
import ImageFiltering
import GaussianMixtures

using Distributions
using LinearAlgebra
using Base.Threads

include("utils.jl")

function tukey_biweight(g_norm, σ_g)
    if(g_norm < σ_g)
        r = g_norm / σ_g
        c = 1 - r*r
        c*c
    else
        0
    end
end

function onad_compute_basis(g_x, g_y)
    g_mag      = sqrt(g_x*g_x + g_y*g_y)
    e0_x, e0_y = if (g_mag > 1e-5)
        g_x / g_mag, g_y / g_mag
    else
        1.0, 0.0
    end
    e1_x = -e0_y
    e1_y = e0_x
    e0_x, e0_y, e1_x, e1_y
end

function onad_compute_coefficient(c_g,
                                  e0_x, e0_y,
                                  e1_x, e1_y,
                                  ctang)
    λ0 = c_g
    λ1 = ctang

    Dxx = λ0*e0_x*e0_x + λ1*e1_x*e1_x
    Dxy = λ0*e0_x*e0_y + λ1*e1_x*e1_y
    Dyy = λ0*e0_y*e0_y + λ1*e1_y*e1_y
    Dxx, Dxy, Dyy
end

function ts_onad(img, dt, n_iters, σ_g, σ_r, ctang)
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)
    img_lpf = Array{Float32}(undef, M, N)

    D_xx = Array{Float32}(undef, M, N)
    D_xy = Array{Float32}(undef, M, N)
    D_yy = Array{Float32}(undef, M, N)
    c_g  = Array{Float32}(undef, M, N)

    ImageFiltering.imfilter!(img_lpf, img_src,
                             ImageFiltering.Kernel.gaussian((4.0, 4.0), (5,5)),
                             "replicate",
                             ImageFiltering.Algorithm.FIR())

    @inbounds for j = 1:N
        @inbounds for i = 1:M
            xm = max(i-1, 1)
            xp = min(i+1, M)
            ym = max(j-1, 1)
            yp = min(j+1, N)

            g_x  = (img_lpf[xp, j]  - img_lpf[xm, j])/2
            g_y  = (img_lpf[i,  yp] - img_lpf[i,  ym])/2

            g_mag    = sqrt(g_x*g_x + g_y*g_y)
            c_g[i,j] = tukey_biweight(g_mag, σ_g)
            e0_x, e0_y, e1_x, e1_y = onad_compute_basis(g_x, g_y)
            d_xx, d_xy, d_yy       = onad_compute_coefficient(c_g[i,j],
                                                              e0_x, e0_y,
                                                              e1_x, e1_y,
                                                              ctang)
            D_xx[i,j] = d_xx
            D_xy[i,j] = d_xy
            D_yy[i,j] = d_yy
        end
    end

    a = D_xx
    b = D_xy
    c = D_yy

    ProgressMeter.@showprogress for t = 1:n_iters
        for j = 1:N
            for i = 1:M
                nx = max(i - 1, 1)
                px = min(i + 1, M)
                ny = max(j - 1, 1)
                py = min(j + 1, N)

                u1 = img_src[nx, py]
                u2 = img_src[i,  py]
                u3 = img_src[px, py]
                u4 = img_src[nx, j ]
                u5 = img_src[i,  j ]
                u6 = img_src[px, j ]
                u7 = img_src[nx, ny]
                u8 = img_src[i,  ny]
                u9 = img_src[px, ny]

                r1 = exp(-img_lpf[nx, py]/σ_r)
                r2 = exp(-img_lpf[i,  py]/σ_r)
                r3 = exp(-img_lpf[px, py]/σ_r)
                r4 = exp(-img_lpf[nx, j ]/σ_r)
                r6 = exp(-img_lpf[px, j ]/σ_r)
                r7 = exp(-img_lpf[nx, ny]/σ_r)
                r8 = exp(-img_lpf[i,  ny]/σ_r)
                r9 = exp(-img_lpf[px, ny]/σ_r)

                A1 = (1/4)*(b[nx, j ] - b[i, py])*r1
                A2 = (1/2)*(c[i,  py] + c[i, j ])*r2
                A3 = (1/4)*(b[px, j ] + b[i, py])*r3
                A4 = (1/2)*(a[nx, j ] + a[i, j ])*r4
                A6 = (1/2)*(a[px, j ] + a[i, j ])*r6
                A7 = (1/4)*(b[nx, j ] + b[i, ny])*r7
                A8 = (1/2)*(c[i,  ny] + c[i, j ])*r8
                A9 = (1/4)*(b[px, j ] - b[i, ny])*r9

                img_dst[i,j] = (u5 + dt*(A1*u1 + A2*u2 + A3*u3 + A4*u4
                                         + A6*u6 + A7*u7 + A8*u8 + A9*u9)) /
                        (1 + dt*(A1 + A2 + A3 + A4 + A6 + A7 + A8 + A9))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end

function ts_onad()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/forearm_gray.png")
    img      = FileIO.load("../data/image/thyroid_add.png")
    #img      = FileIO.load("2.png")
    img      = Images.Gray.(img)
    img      = Float32.(Images.gray.(img))
    img_base = deepcopy(img)

    n_iter = 50
    dt     = 1.0
    σ_g    = 0.1
    σ_r    = 0.1
    ctang  = 1.0

    img_out  = ts_onad(img, dt, n_iter, σ_g, σ_r, ctang)
    img_out  = clamp.(img_out, 0, 1.0)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1, npad=10)
    ImageView.imshow(view)
end

# X_itcpt = np.hstack((X, np.ones((n_data, 1))))
# beta    = np.dot(np.linalg.solve(np.matmul(X_itcpt.T, X_itcpt), X_itcpt.T), y)
# alpha   = beta[n_pred]
