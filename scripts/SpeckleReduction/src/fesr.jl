
function fesr(img, Δt, n_iters, k; mask=trues(size(img)...))
#=

=##
    G_pyramid = Images.gaussian_pyramid(img, 4, 2, 1.0)
    L_pyramid = deepcopy(G_pyramid)

    for i = 1:3
        L_pyramid[i] = G_pyramid[i] - Images.imresize(G_pyramid[i+1], size(G_pyramid[i])...)
    end
    L_pyramid[4] = G_pyramid[4]

    mask_resize  = Images.imresize(mask, size(G_pyramid[1])) .> 0.0
    L_pyramid[1] = pmad(L_pyramid[1], Δt, n_iters, k[1];
                        mask=mask_resize, robust=true)

    mask_resize  = Images.imresize(mask, size(G_pyramid[2])) .> 0.0
    L_pyramid[2] = pmad(L_pyramid[2], Δt, n_iters, k[2];
                        mask=mask_resize, robust=true)

    mask_resize  = Images.imresize(mask, size(G_pyramid[3])) .> 0.0
    σ_L2 = 2.0
    ρ_L2 = 2.0
    α_L2 = 0.1
    β_L2 = 100.0
    imshow(L_pyramid[3])
    L_pyramid[3] = ced(L_pyramid[3], Δt, n_iters, σ_L2, ρ_L2, α_L2, β_L2; mask=mask_resize)
    L_pyramid[3] = clamp.(L_pyramid[3], 0, 255)
    imshow(L_pyramid[3])

    for i = 3:-1:1
        L_pyramid[i] += Images.imresize(L_pyramid[i+1], size(L_pyramid[i])...)
    end
    L_pyramid[1]
end
