
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
