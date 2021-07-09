
using DrWatson
@quickactivate "SpeckleReduction"

include(srcdir("SpeckleReduction.jl"))

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

function ncd_test()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/forearm_gray.png")
    #img      = FileIO.load("2.png")
    #img      = FileIO.load("../data/selections/liver/Test1_1.png")
    img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")
    img      = Images.Gray.(img)
    img      = Float32.(Images.gray.(img))
    img_base = deepcopy(img)

    img_out = ncd(img, 1.0, 30, 3.0, 1.0, 0.5, 0.0001)
    img_out = clamp.(img_out, 0, 1.0)

    FileIO.save("ncd.png", img_out)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1,
                                  npad=10)
    ImageView.imshow(view)
end

function dpad_test()
    #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
    #img      = FileIO.load("../data/image/thyroid_add.png")
    #img      = FileIO.load("../data/selections/liver/Test1_1.png")
    #img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")

    envelop  = PFM.pfmread("../../../data/envelop/v10_convex_liver.pfm")
    img_base = logcompress(envelop, 255, 50)
    img_out  = dpad(envelop, 0.3, 50, 11)
    img_out  = logcompress(img_out, 255, 50)

    #FileIO.save("dpad.png", img_out)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1, npad=10)
    ImageView.imshow(view)
end

function osrad_test()
    envelop  = PFM.pfmread("../../../data/envelop/v10_convex_liver.pfm")
    max_int  = maximum(envelop)
    img_base = logcompress(envelop, max_int, 255, 50)
    img_out  = osrad(envelop, 1.0, 100, 0.1, 21)
    img_out  = logcompress(img_out, max_int, 255, 50)
    
    #FileIO.save("osrad.png", img_out)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1, npad=10)
    ImageView.imshow(view)
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

    PNGFiles.save("posrad.png", img_out)

    view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
                                  ImageCore.colorview(Images.Gray, img_out);
                                  nrow=1,
                                  npad=10)
    ImageView.imshow(view)
end
