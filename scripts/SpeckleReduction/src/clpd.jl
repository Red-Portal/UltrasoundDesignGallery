
function clpd(img,
              ee1_α, ee1_β, ee1_σ,
              ee2_α, ee2_β, ee2_σ,
              ncd1_s, ncd1_α,
              ncd2_s, ncd2_α,
              rpncd_k;
              mask=trues(size(img)...))
#=
    Cascaded Laplacian Pyramid Diffusion
=##
    G     = Images.gaussian_pyramid(img, 3, 2, 1.0)
    L     = deepcopy(G)
    masks = Array{Matrix{Bool}}(undef, 3)
    for i = 1:length(G)-1
        G_scaled = Images.imresize(G[i+1], size(G[i]))
        masks[i] = Images.imresize(mask, size(G[i])) .> 0
        L[i]     = G[i] - G_scaled
    end
    L[end] = G[end]

    f(x, α, β, σ) = begin
        if (abs(x) <= σ)
            sign(x)*σ*(abs(x) / σ)^α
        else
            sign(x)*(β*(abs(x) - σ) + σ)
        end
    end

    G4_up = Images.imresize(L[4], size(G[3]))
    G[3]  = G4_up + L[3]
    #G[3]  = ncd(255*G[3], 2.0, 10, 2.0, ncd1_α, 0.0, ncd1_s; mask=masks[3])/255

    G3_up = Images.imresize(G[3], size(G[2]))
    L[2]  = f.(L[2], ee1_α, ee1_β, ee1_σ)
    G[2]  = G3_up + L[2]
    #G[2]  = ncd(255*G[2], 2.0, 10, 2.0, ncd2_α, 0.0, ncd2_s; mask=masks[2])/255

    G2_up = Images.imresize(G[2], size(G[1]))
    L[1]  = f.(L[1], ee2_α, ee2_β, ee2_σ)
    G[1]  = G2_up + L[1]
    #G[1]  = rpncd(G[1], 0.3, 20, rpncd_k, 5/180*π; mask=masks[1])

    img_out = clamp.(G[1], 0, 1)
end
