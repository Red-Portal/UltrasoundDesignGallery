
import Images
import ImageView
import Plots, StatsPlots
import MosaicViews
import ImageCore
import ProgressMeter
import ImageFiltering
import FileIO

using Distributions
using LinearAlgebra
using Base.Threads

include("utils.jl")
include("dpad.jl")

function osrad_compute_basis(g_x, g_y)
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

function osrad_compute_coefficient(e0_x, e0_y,
                                   e1_x, e1_y,
                                   coeff,
                                   ctang)
    λ0 =  coeff
    λ1 = ctang

    Dxx = λ0*e0_x*e0_x + λ1*e1_x*e1_x
    Dxy = λ0*e0_x*e0_y + λ1*e1_x*e1_y
    Dyy = λ0*e0_y*e0_y + λ1*e1_y*e1_y
    Dxx, Dxy, Dyy
end

function osrad(img, dt, n_iters, ctang, w)
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)
    coeff2  = Array{Float32}(undef, M, N)

    D_xx = Array{Float32}(undef, M, N)
    D_xy = Array{Float32}(undef, M, N)
    D_yy = Array{Float32}(undef, M, N)

    ProgressMeter.@showprogress for t = 1:n_iters
        compute_icov!(img_src, coeff2, w)
        C2_noise = median(coeff2)
        coeff2   = (1 .+ (1 ./ max.(coeff2, 1e-5))) ./ (1 .+ (1 / max.(C2_noise, 1e-5)))

        @inbounds for j = 1:N
            @inbounds for i = 1:M
                xm = max(i-1, 1)
                xp = min(i+1, M)
                ym = max(j-1, 1)
                yp = min(j+1, N)

                g_x = (img[xp, j]  - img[xm, j])/2
                g_y = (img[i,  yp] - img[i,  ym])/2

                e0_x, e0_y, e1_x, e1_y = osrad_compute_basis(g_x, g_y)
                d_xx, d_xy, d_yy       = osrad_compute_coefficient(e0_x, e0_y,
                                                                   e1_x, e1_y,
                                                                   coeff2[i,j],
                                                                   ctang)
                D_xx[i,j] = d_xx
                D_xy[i,j] = d_xy
                D_yy[i,j] = d_yy
            end
        end

        a = D_xx
        b = D_xy
        c = D_yy
        
        @inbounds for j = 1:N
            @inbounds for i = 1:M
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

                A1 = (1/4)*(b[nx, j ] - b[i, py])
                A2 = (1/2)*(c[i,  py] + c[i, j ])
                A3 = (1/4)*(b[px, j ] + b[i, py])
                A4 = (1/2)*(a[nx, j ] + a[i, j ])
                A6 = (1/2)*(a[px, j ] + a[i, j ])
                A7 = (1/4)*(b[nx, j ] + b[i, ny])
                A8 = (1/2)*(c[i,  ny] + c[i, j ])
                A9 = (1/4)*(b[px, j ] - b[i, ny])

                img_dst[i,j] = (u5 + dt*(
                    A1*u1 + A2*u2 + A3*u3
                    + A4*u4 + A6*u6 + A7*u7
                    + A8*u8 + A9*u9)) /
                        (1 + dt*(A1 + A2 + A3
                                 + A4 + A6 + A7
                                 + A8 + A9))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end

function osrad_test()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/forearm_gray.png")
    #img      = FileIO.load("../data/image/thyroid_add.png")
    #img      = FileIO.load("2.png")
    #img      = FileIO.load("../data/selections/liver/Test3_2.png")
    img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")
    img      = Images.Gray.(img)
    img_base = Float32.(Images.gray.(img))
    img      = exp10.(img_base)

    img_out  = osrad(img, 1.0, 100, 0.1, 7)
    img_out  = log10.(clamp.(img_out, 1.0, 10.0))
    
    FileIO.save("osrad.png", img_out)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1, npad=10)
    ImageView.imshow(view)
end
