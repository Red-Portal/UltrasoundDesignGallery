

import Images
import ImageView
import Plots, StatsPlots
import MosaicViews
import ImageCore
import ProgressMeter
import ImageFiltering
import GaussianMixtures
import Noise
import FileIO
import PNGFiles

using Distributions
using LinearAlgebra
using Base.Threads

include("utils.jl")

function fetch_patch!(img, i, j, window_size, patch)
    M      = size(img, 1)
    N      = size(img, 2)
    offset = floor(Int, window_size / 2)
    idx    = 1
    for x = -offset:1:offset
        for y = -offset:1:offset
            patch[idx] = img[clamp(i + x, 1, M), clamp(j + y, 1, N)]
            idx += 1
        end
    end
end

function pixel_relativity(patch_target, patch_origin, u_origin, u_target, σ, m, M)
    u_origin2 = u_origin*u_origin
    u_target2 = u_target*u_target

    m2 = m*m
    M2 = M*M

    r = (m2 + M2)/(u_origin2 + u_target2)*exp(
        -sum((patch_target - patch_origin).^2) / (σ*σ))
    clamp(r, 1.0, 50)
end

function eppr(img, hog_window, pr_window, dt, n_iters, σ)
    M = size(img, 1)
    N = size(img, 2)

    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef,    M, N)
    g_θ     = Array{Float32}(undef,    M, N)
    hog     = Array{Float32}(undef, 8, M, N)

    ProgressMeter.@showprogress for t = 1:n_iters
        @inbounds for j = 1:N
            @inbounds for i = 1:M
                xm = max(i-1, 1)
                xp = min(i+1, M)
                ym = max(j-1, 1)
                yp = min(j+1, N)

                g_x = (img[xp, j]  - img[xm, j])/2
                g_y = (img[i,  yp] - img[i,  ym])/2

                g_θ[i,j] = atan(g_x, g_y)
            end
        end

        hist  = Array{Float32}(undef, 9)
        patch = Array{Float32}(undef, hog_window*hog_window)
        for j = 1:N
            for i = 1:M
                fetch_patch!(g_θ, i, j, hog_window, patch)
                hist[:] .= 0
                for p ∈ patch
                    q = round(Int, (p + π) / π / 2 * 8)
                    hist[q+1] += 1
                end
                hist[1]    += hist[end]
                hist[1:8]  /= sum(view(hist, 1:8))
                hog[:,i,j]  = view(hist, 1:8)
            end
        end

        patch_origin = Array{Float32}(undef, pr_window*pr_window)
        patch_target = Array{Float32}(undef, pr_window*pr_window)
        for j = 1:N
            for i = 1:M
                nx = max(i - 1, 1)
                px = min(i + 1, M)
                ny = max(j - 1, 1)
                py = min(j + 1, N)

                fetch_patch!(img, i, j, pr_window, patch_origin)

                Inp = img_src[nx, py]
                Icp = img_src[i,  py] 
                Ipp = img_src[px, py] 
                Inc = img_src[nx, j] 
                Icc = img_src[i, j] 
                Ipc = img_src[px, j]
                Inn = img_src[nx, ny] 
                Icn = img_src[i,  ny] 
                Ipn = img_src[px, ny]

                fetch_patch!(img, nx, py, pr_window, patch_target)
                Rnp = pixel_relativity(patch_target, patch_origin, Icc, Inp, σ, 0, 1)

                fetch_patch!(img,  i, py, pr_window, patch_target)
                Rcp = pixel_relativity(patch_target, patch_origin, Icc, Icp, σ, 0, 1)

                fetch_patch!(img, px, py, pr_window, patch_target)
                Rpp = pixel_relativity(patch_target, patch_origin, Icc, Ipp, σ, 0, 1)

                fetch_patch!(img, nx,  j, pr_window, patch_target)
                Rnc = pixel_relativity(patch_target, patch_origin, Icc, Inc, σ, 0, 1)

                fetch_patch!(img, px,  j, pr_window, patch_target)
                Rpc = pixel_relativity(patch_target, patch_origin, Icc, Ipc, σ, 0, 1)

                fetch_patch!(img, nx, ny, pr_window, patch_target)
                Rnn = pixel_relativity(patch_target, patch_origin, Icc, Inn, σ, 0, 1)

                fetch_patch!(img,  i, ny, pr_window, patch_target)
                Rcn = pixel_relativity(patch_target, patch_origin, Icc, Icn, σ, 0, 1)

                fetch_patch!(img, px, ny, pr_window, patch_target)
                Rpn = pixel_relativity(patch_target, patch_origin, Icc, Ipn, σ, 0, 1)

                ws  = view(hog, :, i, j)
                wnn = ws[1]*Rnn
                wnc = ws[2]*Rnc
                wnp = ws[3]*Rnp
                wcn = ws[4]*Rcn
                wcp = ws[5]*Rcp
                wpn = ws[6]*Rpn
                wpc = ws[7]*Rpc
                wpp = ws[8]*Rpp

                img_dst[i,j] = (Icc +
                    dt*(Inn*wnn + Icp*wcp
                        + Ipp*wpp + Inp*wnp
                        + Inc*wnc + Ipc*wpc
                        + Icn*wcn + Ipn*wpn)) /
                            (1 + dt*(wnn + wnc + wnp + wcn + wcp + wpn + wpc + wpp))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end


function eppr_test()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/thyroid_add.png")
    #img      = FileIO.load("../data/selections/liver/Test3_2.png")
    img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")
    #img      = FileIO.load("2.png")
    img      = Images.Gray.(img)
    img_base = Float32.(Images.gray.(img))

    img_out = eppr(img_base, 21, 9, 1.0, 10, 3.0)
    img_out = clamp.(img_out, 0.0, 1.0)

    PNGFiles.save("eppr.png", img_out)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1, npad=10)
    ImageView.imshow(view)
end

