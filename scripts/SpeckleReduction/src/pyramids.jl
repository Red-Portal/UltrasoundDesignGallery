
function laplacian_pyramid(img, σ, L)
    pyr = Images.gaussian_pyramid(img, L-1, 2, σ)
    for l = 1:L-1
        pyr[l] -= Images.imresize(pyr[l+1], size(pyr[l]))
    end
    pyr
end
