
import Images
import ImageView
import FileIO
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

function posrad(img, dt, n_iters;
                gmm=nothing,
                n_classes=0,
                mask=nothing,
                fit_freq=5)
    if !isnothing(gmm)
        n_classes = length(gmm.μ)
        fit_freq  = 1
    end

    M = size(img, 1)
    N = size(img, 2)
    C = n_classes

    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)

    img_σ     = Array{Float32}(undef, M, N)
    a         = Array{Float32}(undef, M, N)
    b         = Array{Float32}(undef, M, N)
    c         = Array{Float32}(undef, M, N)
    G_x       = Array{Float32}(undef, M, N, C)
    G_y       = Array{Float32}(undef, M, N, C)
    G_x_σ     = Array{Float32}(undef, M, N, C)
    G_y_σ     = Array{Float32}(undef, M, N, C)
    J_xx_σ    = Array{Float32}(undef, M, N, C)
    J_xy_σ    = Array{Float32}(undef, M, N, C)
    J_yy_σ    = Array{Float32}(undef, M, N, C)
    #G_v_mag   = Array{Float32}(undef, M, N)
    #G_v_mag_σ = Array{Float32}(undef, M, N)

    fuck = Array{Float32}(undef, M, N)

    kernel_x = Images.OffsetArray([3 10  3;  0 0   0; -3 -10 -3]/32.0, -1:1, -1:1)
    kernel_y = Images.OffsetArray([3  0 -3; 10 0 -10;  3   0 -3]/32.0, -1:1, -1:1)

    ρ = 3.0
    k = 5

    ProgressMeter.@showprogress for t = 1:n_iters
        if (mod(t, fit_freq) == 1)
            pixels = reshape(Float64.(img_src[mask]), :)
            pixels = reshape(pixels, (:,1))*100

            if (t == 1)
                gmm = GaussianMixtures.GMM(n_classes, pixels, nIter=0)
            end

            GaussianMixtures.em!(gmm, pixels; nIter=16, varfloor=1e-7)
            idx    = sortperm(gmm.μ[:,1]; rev=true)
            gmm.μ  = gmm.μ[idx,:]
            gmm.Σ  = gmm.Σ[idx,:]
            gmm.w  = gmm.w[idx]
        end

        ImageFiltering.imfilter!(img_σ, img_src,
                                 ImageFiltering.Kernel.gaussian((ρ, ρ), (k,k)),
                                 "replicate",
                                 ImageFiltering.Algorithm.FIR())

        posterior, _ = GaussianMixtures.gmmposterior(gmm, reshape(img_σ, (:,1))*100)
        posterior    = reshape(posterior, (size(img_src)..., size(posterior, 2)))

        for c = 1:n_classes
            G_x[:,:,c] = ImageFiltering.imfilter(
                posterior[:,:,c],
                kernel_x,
                "replicate",
                ImageFiltering.Algorithm.FIR())

            G_y[:,:,c] = ImageFiltering.imfilter(
                posterior[:,:,c], 
                kernel_y,
                "replicate",
                ImageFiltering.Algorithm.FIR())
        end

        G_x_σ = ImageFiltering.imfilter(G_x, 
                                        ImageFiltering.Kernel.gaussian((ρ, ρ, 1), (k, k, 1)),
                                        "replicate",
                                        ImageFiltering.Algorithm.FIR())
        G_y_σ = ImageFiltering.imfilter(G_y, 
                                        ImageFiltering.Kernel.gaussian((ρ, ρ, 1), (k, k, 1)),
                                        "replicate",
                                        ImageFiltering.Algorithm.FIR())

        J_xx = G_x_σ.*G_x_σ
        J_xy = G_x_σ.*G_y_σ
        J_yy = G_y_σ.*G_y_σ

        J_xx_σ = ImageFiltering.imfilter(J_xx, 
                                         ImageFiltering.Kernel.gaussian((ρ, ρ, 1), (k, k, 1)),
                                         "replicate",
                                         ImageFiltering.Algorithm.FIR())
        J_xy_σ = ImageFiltering.imfilter(J_xy, 
                                         ImageFiltering.Kernel.gaussian((ρ, ρ, 1), (k, k, 1)),
                                         "replicate",
                                         ImageFiltering.Algorithm.FIR())
        J_yy_σ = ImageFiltering.imfilter(J_yy, 
                                         ImageFiltering.Kernel.gaussian((ρ, ρ, 1), (k, k, 1)),
                                         "replicate",
                                         ImageFiltering.Algorithm.FIR())

        for j = 1:N
            for i = 1:M
                v1x_k   = 0
                v1y_k   = 0
                v2x_k   = 0
                v2y_k   = 0
                g_v_mag = 0
                λ_k     = -Inf
                for c = 1:C
                    v1x, v1y, v2x, v2y, λ1, _ = eigenbasis_2d(J_xx_σ[i,j,c],
                                                              J_xy_σ[i,j,c],
                                                              J_yy_σ[i,j,c])
                    if (λ1 > λ_k)
                        v1x_k   = v1x
                        v1y_k   = v1y
                        v2x_k   = v2x
                        v2y_k   = v2y
                        λ_k     = λ1
                        g_v_mag = abs(v1x_k*G_x[i,j,c] + v1y_k*G_y[i,j,c])
                        #g_v_mag = sqrt(G_x[i,j,c].^2 + G_y[i,j,c].^2)
                    end
                end

                λ1  = 1.0 .- g_v_mag
                λ2  = 1.0 

                fuck[i,j] = λ1

                a[i,j] = λ1.*v1x_k.*v1x_k + λ2.*v2x_k.*v2x_k
                b[i,j] = λ1.*v1x_k.*v1y_k + λ2.*v2x_k.*v2y_k
                c[i,j] = λ1.*v1y_k.*v1y_k + λ2.*v2y_k.*v2y_k
            end
        end

        display(Plots.histogram(reshape(fuck, :)))

        for j = 1:N
            for i = 1:M
                nx = max(i - 1, 1)
                px = min(i + 1, M)
                ny = max(j - 1, 1)
                py = min(j + 1, N)

                A1 = (1/4)*(b[nx, j ] - b[i, py])
                A2 = (1/2)*(c[i,  py] + c[i, j ])
                A3 = (1/4)*(b[px, j ] + b[i, py])
                A4 = (1/2)*(a[nx, j ] + a[i, j ])
                A6 = (1/2)*(a[px, j ] + a[i, j ])
                A7 = (1/4)*(b[nx, j ] + b[i, ny])
                A8 = (1/2)*(c[i,  ny] + c[i, j ])
                A9 = (1/4)*(b[px, j ] - b[i, ny])

                img_dst[i,j] = (img_src[i,j] + dt*(
                    A1*(img_src[nx, py]) + 
                    A2*(img_src[i,  py]) + 
                    A3*(img_src[px, py]) + 
                    A4*(img_src[nx, j])  + 
                    A6*(img_src[px, j])  + 
                    A7*(img_src[nx, ny]) + 
                    A8*(img_src[i,  ny]) + 
                    A9*(img_src[px, ny])))  /
                    (1 + dt*(A1 + A2 + A3 + A4 + A6 + A7 + A8 + A9))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end

function fit_pixel_gmm(target)
    gmm_file = if target == :liver
        "liver_gmm.jld2"
    else
        "thyroid_gmm.jld2"
    end

    if(isfile(gmm_file))
        FileIO.load(gmm_file, "gmm")
    else
        mask = if target == :liver
            FileIO.load("convex_mask.png") 
        else
            FileIO.load("linear_mask.png") 
        end
        mask = Images.Gray.(mask) 
        mask = Float32.(mask) .> 0

        n_classes = if target == :liver
            3
        else
            4
        end

        fnames = if target == :liver
            ["../data/subjects/thyroid/Test1_1.png",
             "../data/subjects/thyroid/Test1_2.png",
             "../data/subjects/thyroid/Test1_3.png",
             "../data/subjects/thyroid/Test1_4.png",
             "../data/subjects/thyroid/Test2_1.png",
             "../data/subjects/thyroid/Test2_2.png",
             "../data/subjects/thyroid/Test2_3.png",
             "../data/subjects/thyroid/Test2_4.png",
             ]
        else
            ["../data/subjects/thyroid/m_min__000000.jpg",
             "../data/subjects/thyroid/m_min__000003.jpg",
             "../data/subjects/thyroid/m_min__000005.jpg",
             "../data/subjects/thyroid/m_min__000007.jpg",
             "../data/subjects/thyroid/m_min__000010.jpg",
             "../data/subjects/thyroid/m_PJH_000000.jpg",
             "../data/subjects/thyroid/m_PJH_000003.jpg",
             "../data/subjects/thyroid/m_PJH_000005.jpg",
             "../data/subjects/thyroid/m_PJH_000007.jpg",
             "../data/subjects/thyroid/m_PJH_000010.jpg",
             ]
        end

        pixels = mapreduce(vcat, fnames) do fname
            img = FileIO.load(fname)
            img = Float64.(Images.Gray.(img))
            reshape(img[mask], :)
        end
        gmm   = GaussianMixtures.GMM(n_classes, pixels, nIter=0)
        idx   = sortperm(gmm.μ[:,1]; rev=true)
        gmm.μ = gmm.μ[idx,:]
        gmm.Σ = gmm.Σ[idx,:]
        gmm.w = gmm.w[idx]

        GaussianMixtures.em!(gmm, reshape(pixels, (:,1)); nIter=16, varfloor=1e-5)
        FileIO.save(gmm_file, "gmm", gmm)
        gmm 
    end
end

function posrad_test()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/forearm_gray.png")
    #img      = FileIO.load("2.png")
    #img      = FileIO.load("../data/selections/liver/Test3_2.png")
    img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")
    img      = Images.Gray.(img)
    img      = Float32.(img)
    img_base = deepcopy(img)

    target   = :thyroid
    #gmm      = fit_pixel_gmm(target)
    mask     = if target == :liver
        FileIO.load("convex_mask.png") 
    else
        FileIO.load("linear_mask.png") 
    end
    mask = Images.Gray.(mask) 
    mask = Float32.(mask) .> 0

    dt        = 0.5
    n_iters   = 50
    n_classes = 4
    img_out   = posrad(img, dt, n_iters, gmm=nothing, mask=mask, n_classes=n_classes)
    img_out   = clamp.(img_out, 0, 1.0)

    FileIO.save("posrad.png", img_out)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1,
                                  npad=10)
    ImageView.imshow(view)
end
