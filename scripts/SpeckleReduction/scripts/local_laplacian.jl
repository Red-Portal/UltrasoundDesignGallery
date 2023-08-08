
using DrWatson
@quickactivate "SpeckleReduction"

include(srcdir("SpeckleReduction.jl"))
include("evaluate_algorithms.jl")

import ImageView
import MosaicViews
import PNGFiles
import ProgressMeter

function main()
    #img = PFM.pfmread("cardiac1_1.pfm")
    #img = PFM.pfmread("../../../data/selections/cardiac3/cardiac6_50.pfm")
    #img = PFM.pfmread("../../../data/selections/cardiac1/cardiac1_2.pfm")
    #img = PFM.pfmread("../../../data/selections/liver2/liver4_1.pfm")
    #img = PNGFiles.load("thyroid_add.png")
    #img = PNGFiles.load("hansono4.png")
    img = FileIO.load("moai.jpg")

    img = Float32.(Images.Gray.(img))

    #img = PNGFiles.load("liver.png")*255
    #img = Images.imresize(img, (300, 400))

    #mask = PNGFiles.load("../../../data/selections/liver2/liver4_1.png") .> 0.0
    #mask = PNGFiles.load("../../../data/selections/cardiac3/cardiac6_1.png") .> 0.0
    #mask = trues(size(img))

    #return ImageView.imshow( Images.imresize(pyr[2], size(img)) - img )
    #return ImageView.imshow( Images.imresize(pyr[3], size(img)) - img )

    #img_res = cshock(img*255, 0.2, 40, 0.1, 0.1, 0.9, 0.01)/255
    #img_res = musica(img*255, [1.0, 2.0, 2.0], [0.1, 1.2, 1.2], 50, 2.0, 4)/255 
    #img_res = run_mnlm(img, mask)
    #img_res = rpncd(img*255, 0.5, 40, 2.0, 5/180*π)/255
    #img_res = ncd(img*255, 2.0, 20, 2.0, 0.05, 0.0, 5)/255

    # Liver
    #img_res = adjust_dynamic_range(img_res, 0.15, 0.5)

    # Thyroid
    #img_res = adjust_dynamic_range(img_res, 0.05, 0.7)

    #PNGFiles.save("/home/msca8h/Documents/seminars/22-hansono/figures/liver_rpncd_largek.png", img_res)
    #PNGFiles.save("/home/msca8h/Documents/seminars/22-hansono/figures/cardiac_mnlm.png", img_res)
    #PNGFiles.save("/home/msca8h/Documents/seminars/22-hansono/figures/thyroid_mnlm.png", img_res)
    #PNGFiles.save("/home/msca8h/Documents/seminars/22-hansono/figures/forearm_musica.png", img_res)

    #return ImageView.imshow(img_res)

    N   = 15
    L   = 4
    #σ   = 2.0
    #α   = -1.1
    #β   = 1.0
    σ   = 2.0
    #α   = 10.0
    β   = 1.2
    α   = -1.0
    σ_r = 30
    L   = fllf(img*255, σ, σ_r, α, β, L, N)
    #L   = gllf(img, α, σ, σ_r, 20, L, 255)

    #L[4]  = cshock(L[4], 0.2, 20, 0.1, 0.1, 0.0, 0.01)

    L[3]  = L[3] + Images.imresize(L[4], size(L[3]))
    L[3]  = ncd(L[3], 2.0, 40, 2.0, 0.1, 0.0, 0)
    #L3_old = deepcopy(L[3])
    #L[3]  = shock(L[3], 0.1, 3.0, 3.0, 5)
    #L[3]  = cshock(L[3], 0.2, 30, 0.1, 0.03, 0.9, 0.01)

    L[2] = L[2] + Images.imresize(L[3], size(L[2]))
    L[2] = ncd(L[2], 2.0, 40, 2.0, 0.1, 0.0, 0)

    L[1] = L[1] + Images.imresize(L[2], size(L[1]))
    #L[1] = ncd(L[1], 2.0, 20, 2.0, 0.01, 0.0, 5)
    #L[1] = rpncd(L[1], 0.3, 20, 0.1, 5/180*π)
    # L[1]  = cshock(L[1], 0.2, 40, 0.1, 0.1, 0.9, 0.01)
    #L[1]  = shock(L[1], 0.1, 3.0, 3.0, 5)

    #pyr  = Images.gaussian_pyramid(img*255, 3, 2, 2.0)
    #L[1] = pyr[1]
    #L[1] = rpncd(L[1], 0.3, 40, 1.0, 5/180*π)

    # kx, ky = ImageFiltering.Kernel.sobel()
    # kσ     = ImageFiltering.Kernel.gaussian(2.0)
    # img_σ  = ImageFiltering.imfilter(L[1], kσ)
    # Gx     = ImageFiltering.imfilter(img_σ, kx)
    # Gy     = ImageFiltering.imfilter(img_σ, ky)

    img_res = clamp.(L[1], 0, 255)/255


    img     = adjust_dynamic_range(img, 0.05, 0.5)
    img_res = adjust_dynamic_range(img_res, 0.05, 0.5)

    #view = MosaicViews.mosaicview(img, img_res, (img - img_res) .+ 0.5; nrow=1)
    view = MosaicViews.mosaicview(img, img_res; nrow=1)
    return ImageView.imshow(view)

    #img_res = adjust_dynamic_range(img_res, 0.05, 0.5)
    #img_res = adjust_dynamic_range(img_res, 0.15, 0.5)
    #ImageView.imshow(img_res)
    # ImageView.imshow(img_res)

    #PNGFiles.save("/home/msca8h/Documents/seminars/22-hansono/figures/cardiac_ncdl3.png", img_res)
    #PNGFiles.save("/home/msca8h/Documents/seminars/22-hansono/figures/forearm_llf2.png", img_res)
    #PNGFiles.save("/home/msca8h/Documents/seminars/22-hansono/figures/cardiac_cfs.png", img_res)
    #PNGFiles.save("/home/msca8h/Documents/seminars/22-hansono/figures/thyroid_llf2.png", img_res)
end

