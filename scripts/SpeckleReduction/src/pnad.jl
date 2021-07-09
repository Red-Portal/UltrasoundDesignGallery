
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
import PNGFiles

using Distributions
using LinearAlgebra
using Base.Threads

include("utils.jl")

function pnad(img, gmm, dt, n_iters)
    n_classes = length(gmm.μ)

    M = size(img, 1)
    N = size(img, 2)
    C = n_classes

    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)

    img_σ     = Array{Float32}(undef, M, N)
    a         = zeros(M, N)
    b         = zeros(M, N)
    c         = zeros(M, N)

    Dxx       = zeros(M, N, C)
    Dxy       = zeros(M, N, C)
    Dyy       = zeros(M, N, C)

    G_x       = Array{Float32}(undef, M, N, C)
    G_y       = Array{Float32}(undef, M, N, C)
    G_x_σ     = Array{Float32}(undef, M, N, C)
    G_y_σ     = Array{Float32}(undef, M, N, C)

    ρ = 1.0
    k = 3 

    ImageFiltering.imfilter!(img_σ, img_src,
                             ImageFiltering.Kernel.gaussian((ρ, ρ), (k,k)),
                             "replicate",
                             ImageFiltering.Algorithm.FIR())

    posterior, _ = GaussianMixtures.gmmposterior(gmm, reshape(img_σ, (:,1)))
    posterior    = reshape(posterior, (size(img_src)..., size(posterior, 2)))

    @inbounds for c = 1:C
        @inbounds for j = 1:N
            @inbounds for i = 1:M
                xm = max(i-1, 1)
                xp = min(i+1, M)
                ym = max(j-1, 1)
                yp = min(j+1, N)

                G_x[i,j,c] = (posterior[xp, j,  c] - posterior[xm, j,  c])/2
                G_y[i,j,c] = (posterior[i,  yp, c] - posterior[i,  ym, c])/2
            end
        end
    end

    G_x_σ = ImageFiltering.imfilter(G_x, 
                                    ImageFiltering.Kernel.gaussian((ρ, ρ, 1), (k, k, 1)),
                                    "replicate",
                                    ImageFiltering.Algorithm.FIR())
    G_y_σ = ImageFiltering.imfilter(G_y, 
                                    ImageFiltering.Kernel.gaussian((ρ, ρ, 1), (k, k, 1)),
                                    "replicate",
                                    ImageFiltering.Algorithm.FIR())

    G_mag   = sqrt.(G_x.*G_x + G_y.*G_y)

    c_all  = [0.00, 0.1, 1.0, 1.0]
    #λ1_all = zeros(C)
    for j = 1:N
        for i = 1:M
            for c = 1:C
                p   = posterior[i, j, c]
                λ1  = tukey_biweight(G_mag[i,j,c], c_all[c])*p
                λ2  = c_all[c]*p

                v1x = G_x_σ[i,j,c]
                v1y = G_y_σ[i,j,c]

                v_mag = sqrt(v1x*v1x + v1y*v1y)
                if (v_mag > 1e-7)
                    v1x /= v_mag
                    v1y /= v_mag
                else
                    v1x = 1.0
                    v1y = 0.0
                end
                v2x = -v1y
                v2y =  v1x

                Dxx[i,j,c] += λ1.*v1x.*v1x + λ2.*v2x.*v2x
                Dxy[i,j,c] += λ1.*v1x.*v1y + λ2.*v2x.*v2y
                Dyy[i,j,c] += λ1.*v1y.*v1y + λ2.*v2y.*v2y
            end
        end
    end

    a = mean(Dxx, dims=3)[:,:,1]
    b = mean(Dxy, dims=3)[:,:,1]
    c = mean(Dyy, dims=3)[:,:,1]

    ProgressMeter.@showprogress for t = 1:n_iters
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
        GaussianMixtures.em!(gmm, reshape(pixels, (:,1)); nIter=16, varfloor=1e-5)
        idx   = sortperm(gmm.μ[:,1]; rev=true)
        gmm.μ = gmm.μ[idx,:]
        gmm.Σ = gmm.Σ[idx,:]
        gmm.w = gmm.w[idx]
        FileIO.save(gmm_file, "gmm", gmm)
        gmm 
    end
end

function pnad_test()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/forearm_gray.png")
    #img      = FileIO.load("2.png")
    #img      = FileIO.load("../data/selections/liver/Test3_2.png")
    img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")
    img      = Images.Gray.(img)
    img      = Float32.(img)
    img_base = deepcopy(img)

    target   = :thyroid
    gmm      = fit_pixel_gmm(target)

    dt        = 1.0
    n_iters   = 50
    img_out   = pnad(img, gmm, dt, n_iters)
    img_out   = clamp.(img_out, 0, 1.0)
    PNGFiles.save("ponad.png", img_out)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1,
                                  npad=10)
    ImageView.imshow(view)
end
