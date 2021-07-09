
import Images
import ImageView
import Plots, StatsPlots
import MosaicViews
import ImageCore
import ProgressMeter
import ImageFiltering
import GaussianMixtures
import FileIO

using Distributions
using LinearAlgebra
using Base.Threads

include("utils.jl")

function shock(img, dt, ρ, σ, n_iters)
    kernel_x = Images.OffsetArray([3 10  3;  0 0   0; -3 -10 -3]/32.0, -1:1, -1:1)
    kernel_y = Images.OffsetArray([3  0 -3; 10 0 -10;  3   0 -3]/32.0, -1:1, -1:1)

    M = size(img, 1)
    N = size(img, 2)

    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)
    Δu      = Array{Float32}(undef, M, N)


    k_ρ = floor(Int, ρ*6/2)*2 + 1
    k_σ = floor(Int, σ*6/2)*2 + 1
    img_σ = ImageFiltering.imfilter(
        img_src,
        ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ)),
        "replicate",
        ImageFiltering.Algorithm.FIR())

    v_x = ImageFiltering.imfilter(
        img_σ,
        kernel_x,
        "replicate",
        ImageFiltering.Algorithm.FIR())
    v_y = ImageFiltering.imfilter(
        img_σ,
        kernel_y,
        "replicate",
        ImageFiltering.Algorithm.FIR())

    v_xx = ImageFiltering.imfilter(
        v_x,
        kernel_x,
        "replicate",
        ImageFiltering.Algorithm.FIR())
    v_xy = ImageFiltering.imfilter(
        v_x,
        kernel_y,
        "replicate",
        ImageFiltering.Algorithm.FIR())
    v_yy = ImageFiltering.imfilter(
        v_y,
        kernel_y,
        "replicate",
        ImageFiltering.Algorithm.FIR())

    ProgressMeter.@showprogress for t = 1:n_iters
        u_x = ImageFiltering.imfilter(
            img,
            kernel_x,
            "replicate",
            ImageFiltering.Algorithm.FIR())
        u_y = ImageFiltering.imfilter(
            img,
            kernel_y,
            "replicate",
            ImageFiltering.Algorithm.FIR())

        J_xx_σ = ImageFiltering.imfilter(
            u_x .* u_x,
            ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
            "replicate",
            ImageFiltering.Algorithm.FIR())
        J_xy_σ = ImageFiltering.imfilter(
            u_x .* u_y,
            ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
            "replicate",
            ImageFiltering.Algorithm.FIR())
        J_yy_σ = ImageFiltering.imfilter(
            u_y .* u_y,
            ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
            "replicate",
            ImageFiltering.Algorithm.FIR())

        for j = 1:N
            for i = 1:M
                v1x, v1y, v2x, v2y, λ1, _ = eigenbasis_2d(J_xx_σ[i,j],
                                                          J_xy_σ[i,j],
                                                          J_yy_σ[i,j])
                w_x = v1x
                w_y = v1y
                Δu = -sign(v_xx[i,j]*w_x*w_x + v_xy[i,j]*w_x*w_y + v_yy[i,j]*w_y*w_y)*
                    sqrt(u_x[i,j].^2 + u_y[i,j].^2)

                img_dst[i,j] = img_src[i,j] + dt*Δu
            end
        end
    end
    img_dst
end

include("dpad.jl")

function shock_test()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/forearm_gray.png")
    #img      = FileIO.load("2.png")
    #img      = FileIO.load("../data/selections/liver/Test3_2.png")
    #img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")
    img      = FileIO.load("ncd.png")
    img      = Images.Gray.(img)
    img      = Float32.(img)
    img_base = deepcopy(img)

    n_iters   = 20
    img_out   = shock(img, 4.0, 1.0, 0.1, n_iters)
    img_out   = clamp.(img_out, 0, 1.0)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1,
                                  npad=10)
    ImageView.imshow(view)
end
