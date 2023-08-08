
function fllf(img, σ, σ_r, α, β, L, N; I_min = 0.0, I_max = 255.0)
#=
    Fast local Laplacian pyramid filter

    M. Aubry, S. Paris, S. Hasinoff, J. Kautz, F. Durand,
    "Fast local Laplacian filters: theory and applications."
    in ACM Transactions on Graphics, 2014.
=##
    pyramid_G        = Images.gaussian_pyramid(img, L-1, 2, σ)
    pyramid_L        = map(deepcopy, pyramid_G)
    pyramids_L_quant = Array{typeof(pyramid_G)}(undef, N)
    I_range          = I_max - I_min

    #fuck = Array{typeof(pyramid_G[1])}(undef, N)
    for n = 0:N-1
        g        = I_min + n/(N-1)*I_range
        # remap(i) = if abs(i - g) ≤ σ_r
        #     g + sign(i - g)*σ_r*(abs(i - g)/σ_r)^α
        # else
        #     g + sign(i - g)*(β*(abs(i - g) - σ_r) + σ_r)
        # end
        remap(i) = (i - g)*(α*exp((i - g)^2 / (-2*σ_r^2))*β + (β - 1)) + i
        I_remap  = map(remap, img)
        pyramids_L_quant[n+1] = laplacian_pyramid(I_remap, σ, L)
    end

    quantize_index(i) = begin
        Δi    = I_range/(N - 1)
        q_idx = floor(Int, i/Δi)
        if q_idx < 0
            q_l = 0
            q_h = 0
            α   = 1.0
            q_l, q_h, α
        elseif q_idx + 1 > N - 1
            q_l = N - 1
            q_h = N - 1
            α   = 0.0
            q_l, q_h, α
        else
            q_l = q_idx
            q_h = q_idx + 1
            α   = 1 - (i - (q_idx*Δi))/Δi
            q_l, q_h, α
        end
    end

    for l = 0:L-1
        pyramid_L[l+1] = map(CartesianIndices(pyramid_G[l+1])) do idx
            q_l, q_h, α = quantize_index(pyramid_G[l+1][idx])
            l_l         = pyramids_L_quant[q_l+1][l+1][idx]
            l_h         = pyramids_L_quant[q_h+1][l+1][idx]
            α*l_l + (1 - α)*l_h
        end
    end
    pyramid_L[end] = pyramid_G[end]
    pyramid_L
end
