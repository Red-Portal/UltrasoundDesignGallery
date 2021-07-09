
import Images
import ImageView
import FileIO
import Plots, StatsPlots
import MosaicViews
import ImageCore
import ProgressMeter
import ImageFiltering
import PFM

using Distributions
using LinearAlgebra
using Base.Threads

include("utils.jl")

function compute_icov!(img_src, coeff2_dst, w)
    M = size(img_src, 1)
    N = size(img_src, 2)

    Ex  = ImageFiltering.mapwindow(mean, img_src,          (w, w))
    Ex2 = ImageFiltering.mapwindow(mean, img_src.*img_src, (w, w))
    var = Ex2 - Ex.*Ex
    
    coeff2_dst[:,:] = var ./ max.(Ex.*Ex, 1e-5)
end

function dpad(img, dt, n_iters, w)
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)
    coeff2  = Array{Float32}(undef, M, N)

    ProgressMeter.@showprogress for t = 1:n_iters
        compute_icov!(img_src, coeff2, w)
        C2_noise = median(coeff2)
        coeff2   = (1 .+ (1 ./ max.(coeff2, 1e-7))) ./ (1 .+ (1 ./ max.(C2_noise, 1e-7)))

        @inbounds for j = 1:N
            @inbounds for i = 1:M
                xp = min(i+1, M)
                xm = max(i-1, 1)
                yp = min(j+1, N)
                ym = max(j-1, 1)

                c_xm = coeff2[xm,  j]
                c_xp = coeff2[xp,  j]
                c_yp = coeff2[i,  yp]
                c_ym = coeff2[i,  ym]

                u_xp = img_src[xp,  j]
                u_xm = img_src[xm,  j]
                u_yp = img_src[i,  yp]
                u_ym = img_src[i,  ym]

                img_dst[i,j] = (img_src[i,j] + dt*(
                    u_xp*c_xp + u_xm*c_xm + u_yp*c_yp + u_ym*c_ym)) / 
                        (1 + dt*(c_xp + c_xm + c_yp + c_ym))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end

function dpad_test()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/thyroid_add.png")
    #img      = FileIO.load("../data/selections/liver/Test1_1.png")
    #img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")

    envelop  = PFM.pfmread("../data/envelop/v10_convex_liver.pfm")
    img_base = logcompress(envelop, 255, 50)
    img_out  = dpad(envelop, 0.3, 50, 11)
    img_out  = logcompress(img_out, 255, 50)

    #FileIO.save("dpad.png", img_out)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1, npad=10)
    ImageView.imshow(view)
end
