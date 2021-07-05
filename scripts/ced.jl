
import Images
import ImageView
import FileIO
import Plots
import MosaicViews
import ImageCore
import ProgressMeter
import ImageFiltering
using Base.Threads

include("utils.jl")

function ced(img, dt, n_iters, σ, ρ, α, β)
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)

    a      = Array{Float32}(undef, M, N)
    b      = Array{Float32}(undef, M, N)
    c      = Array{Float32}(undef, M, N)
    J_xx   = Array{Float32}(undef, M, N)
    J_xy   = Array{Float32}(undef, M, N)
    J_yy   = Array{Float32}(undef, M, N)
    J_xx_σ = Array{Float32}(undef, M, N)
    J_xy_σ = Array{Float32}(undef, M, N)
    J_yy_σ = Array{Float32}(undef, M, N)
    img_σ  = Array{Float32}(undef, M, N)

    k_ρ = floor(Int, ρ*6/2)*2 + 1
    k_σ = floor(Int, σ*6/2)*2 + 1
    ProgressMeter.@showprogress for t = 1:n_iters
        ImageFiltering.imfilter!(img_σ, img_src,
                                 ImageFiltering.Kernel.gaussian((σ, σ), (k_σ,k_σ)),
                                 "replicate",
                                 ImageFiltering.Algorithm.FIR())

        for j = 1:N
            for i = 1:M
                I_xp = img_σ[min(i+1, M), j]
                I_xm = img_σ[max(i-1, 1), j]
                I_yp = img_σ[i, min(j+1, N)]
                I_ym = img_σ[i, max(j-1, 1)]

                g_x = (I_xp - I_xm)/2
                g_y = (I_yp - I_ym)/2

                J_xx[i,j] = g_x*g_x
                J_xy[i,j] = g_x*g_y
                J_yy[i,j] = g_y*g_y
            end
        end

        ImageFiltering.imfilter!(J_xx_σ, J_xx, 
                                 ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
                                 "replicate",
                                 ImageFiltering.Algorithm.FIR())
        ImageFiltering.imfilter!(J_xy_σ, J_xy,
                                 ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
                                 "replicate",
                                 ImageFiltering.Algorithm.FIR())
        ImageFiltering.imfilter!(J_yy_σ, J_yy,
                                 ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
                                 "replicate",
                                 ImageFiltering.Algorithm.FIR())

        for j = 1:N
            for i = 1:M
                v1x, v1y, v2x, v2y, λ1, λ2 = eigenbasis_2d(J_xx_σ[i,j],
                                                           J_xy_σ[i,j],
                                                           J_yy_σ[i,j])

                κ  = (λ1 - λ2).^2
                λ1 = α
                λ2 = if abs(κ) > eps(Float32)
                    α + (1 - α)*exp(-β ./ κ)
                else
                    α 
                end

                Dxx = λ1*v1x*v1x + λ2*v2x*v2x
                Dxy = λ1*v1x*v1y + λ2*v2x*v2y
                Dyy = λ1*v1y*v1y + λ2*v2y*v2y

                a[i,j] = Dxx;
                b[i,j] = Dxy;
                c[i,j] = Dyy;
            end
        end

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

function ced_test()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/forearm_gray.png")
    #img      = FileIO.load("2.png")
    #img      = FileIO.load("../data/selections/liver/Test1_1.png")
    img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")
    img      = Images.Gray.(img)
    img      = Float32.(Images.gray.(img))*255
    img_base = deepcopy(img)

    img_out = ced(img, 1.0, 30, 0.5, 4.0, 0.1, 1.0)
    img_out = clamp.(img_out, 0, 255.0)

    #FileIO.save("ced.png", img_out/255.0)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1,
                                  npad=10)
    ImageView.imshow(view)
end
