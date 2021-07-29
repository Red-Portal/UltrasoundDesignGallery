
function susan(img, r::Int, T::Real)
    n_max = ceil(r*r*π)
    gt    = 3*n_max/4

    img_padded   = Images.padarray(img, Images.Pad(r, r))
    edge_out     = zeros(size(img))
    window_idx   = CartesianIndex(ntuple(i -> r, 2))
    window       = LocalFilters.RectangularBox{2}(-window_idx, window_idx)
    k_min, k_max = LocalFilters.limits(window)

    for p in LocalFilters.cartesianregion(img)
        n = 0.0
        @inbounds for q in LocalFilters.cartesianregion(p - k_max, p - k_min)
            Δidx = Tuple(p - q)
            dist = sqrt(Δidx[1]*Δidx[1] + Δidx[2]*Δidx[2])
            if (dist > r)
                continue
            end
            ΔI = img_padded[q] - img_padded[p]
            n += exp(-(ΔI / T)^6)
        end
        edge_out[p] = if n < gt
            (gt - n) / n_max
        else
            0
        end
    end
    return edge_out
end

function susan_pm(img, Δt, n_iters, σ, susan_r, susan_T;
             mask=trues(size(img)...))
#=
    SUSAN-Controlled Anisotropic Diffusion

    "Ultrasound speckle reduction by a SUSAN-controlled anisotropic diffusion method"
    Jinhua Yu, Jinglu Tan, Yuanyuan Wang
    Pattern Recognition, 2010
=##
    M = size(img, 1)
    N = size(img, 2)

    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)
    G_mag   = zeros(Float32, M, N)

    C               = 1.4826
    k_σ             = floor(Int, max(σ, 1)*6/2)*2 + 1
    smooth_σ_kernel = ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ))
    filter_type     = ImageFiltering.Algorithm.FIR()
    border_type     = "replicate"

    ProgressMeter.@showprogress for t = 1:n_iters
        img_src_σ = ImageFiltering.imfilter(img_src, smooth_σ_kernel, border_type, filter_type)
        G_mag     = susan(img_src_σ, susan_r, susan_T)
        k         = C*median(abs.(G_mag[mask] .- median(G_mag[mask])))
        k         = max(k, 1e-3)
        coeff     = exp.(- (G_mag ./ (4*k*k)).^2)

        for j = 1:N
            for i = 1:M
                if !mask[i,j]
                    continue
                end

                I_c = img_src[i,  j]
                I_w = fetch_pixel(img_src, i-1,   j, mask, I_c)
                I_e = fetch_pixel(img_src, i+1,   j, mask, I_c)
                I_n = fetch_pixel(img_src,   i, j-1, mask, I_c)
                I_s = fetch_pixel(img_src,   i, j+1, mask, I_c)

                C_c = coeff[i,j]
                C_w = fetch_pixel(coeff, i-1,   j, mask, C_c)
                C_e = fetch_pixel(coeff, i+1,   j, mask, C_c)
                C_n = fetch_pixel(coeff,   i, j-1, mask, C_c)
                C_s = fetch_pixel(coeff,   i, j+1, mask, C_c)

                img_dst[i,j] = (I_c +
                    Δt*(C_w*I_w + C_e*I_e + C_n*I_n + C_s*I_s)) /
                    (1 + Δt*(C_w + C_e + C_n + C_s))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end
