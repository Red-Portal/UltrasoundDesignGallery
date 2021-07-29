
function guided_aeed(img, guide, Δt, n_iters, σ, k, K;
                    mask=trues(size(img)...))
    M = size(img, 1)
    N = size(img, 2)

    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)
    G_x     = zeros(Float32, M, N)
    G_y     = zeros(Float32, M, N)
    G_x_σ   = zeros(Float32, M, N)
    G_y_σ   = zeros(Float32, M, N)
    D_xx    = zeros(Float32, M, N)
    D_xy    = zeros(Float32, M, N)
    D_yy    = zeros(Float32, M, N)

    k_σ             = floor(Int, max(σ, 1)*6/2)*2 + 1
    smooth_σ_kernel = ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ))
    filter_type     = ImageFiltering.Algorithm.FIR()
    border_type     = "replicate"

    @inbounds for j = 1:N
        @inbounds for i = 1:M
            if !mask[i,j]
                continue
            end

            I_c = guide[i,  j]
            I_w = fetch_pixel(guide, i-1,   j, mask, I_c)
            I_e = fetch_pixel(guide, i+1,   j, mask, I_c)
            I_n = fetch_pixel(guide,   i, j-1, mask, I_c)
            I_s = fetch_pixel(guide,   i, j+1, mask, I_c)

            G_x[i,j]   = (I_w - I_e)/2
            G_y[i,j]   = (I_n - I_s)/2
        end
    end

    ImageFiltering.imfilter!(G_x_σ, G_x, smooth_σ_kernel, border_type, filter_type)
    ImageFiltering.imfilter!(G_y_σ, G_y, smooth_σ_kernel, border_type, filter_type)

    G_mag² = G_x_σ.*G_x_σ + G_y_σ.*G_y_σ
    α      = 2 .- (2 ./ (1 .+ k*G_mag²))

    #display(Plots.histogram(α[mask]))

    @inbounds for j = 1:N
        @inbounds for i = 1:M
            if !mask[i,j]
                continue
            end
            v1x   = G_x_σ[i,j]
            v1y   = G_y_σ[i,j]
            v_mag = sqrt.(v1x*v1x + v1y*v1y)
            if (v_mag > 1e-5)
                v1x /= v_mag
                v1y /= v_mag
            else
                v1x = 1.0
                v1y = 0.0
            end
            v2x = v1y
            v2y = -v1x

            s  = abs(G_x[i,j].*v1x + G_y[i,j].*v2x)
            λ1 = 1 / (1 + (s/K).^α[i,j])
            λ2 = 1.0

            D_xx[i,j] = λ1*v1x*v1x + λ2*v2x*v2x
            D_xy[i,j] = λ1*v1x*v1y + λ2*v2x*v2y
            D_yy[i,j] = λ1*v1y*v1y + λ2*v2y*v2y
        end
    end

    ProgressMeter.@showprogress for t = 1:n_iters
        weickert_matrix_diffusion!(img_dst, img_src, Δt, D_xx, D_xy, D_yy; mask=mask)
        @swap!(img_src, img_dst)
    end
    img_dst
end

function butterworth(M, N, n_order::Int, cutoff::Real)
    M_offset = M / 2
    N_offset = N / 2
    D0       = cutoff/2
    kernel   = zeros(M, N)
    for idx ∈ CartesianIndices(kernel)
        θ_x         = (idx[1] - M_offset) / M
        θ_y         = (idx[2] - N_offset) / N
        D           = sqrt(θ_x*θ_x + θ_y*θ_y)
        kernel[idx] = 1 / (1 + (D/D0).^(2*n_order))
    end
    kernel
end

function msamd(img, Δt, n_iters, ratio; mask=trues(size(img)...))
#=
    Multi-Scale Adaptive Matrix Diffusion
=##

    M_pad  = nextpow(2, size(img, 1))
    N_pad  = nextpow(2, size(img, 2))
    padded = zeros(M_pad, N_pad)
    θ_cut  = min(size(img, 1), size(img, 2)) / min(M_pad, N_pad) * ratio

    padded[1:size(img,1), 1:size(img,2)] = img
    kernel = butterworth(M_pad, N_pad, 4, θ_cut)
    kernel = FFTW.ifftshift(kernel)
    F      = FFTW.fft(padded, (1,2))
    F_lp   = F .* kernel
    img_lp = FFTW.ifft(F_lp, (1:2))
    img_lp = abs.(img_lp[1:size(img, 1), 1:size(img,2)])

    G0 = img
    G1 = Images.imresize(img_lp, ratio=ratio)
    L0 = G0 - img_lp
    L1 = G1

    #ImageView.imshow(L0, name="L0")
    #ImageView.imshow(G0, name="G0")
    #ImageView.imshow(L1, name="L1")
    #ImageView.imshow(G1, name="G1")

    mask0 = mask
    mask1 = Images.imresize(mask, size(L1)) .> 0.0

    Δt      = 1.0
    n_iters = 30
    σ       = 1.0
    k       = 100.0
    K       = 0.01
    #L0 = guided_apm(L0, G0, Δt, n_iters, σ, k, K; mask=mask0)
    L0 = guided_aeed(L0, G0, Δt, n_iters, σ, k, K; mask=mask0)

    Δt      = 1.0
    n_iters = 20
    σ       = 1.0
    k       = 100.0
    K       = 0.01
    #L1 = guided_apm(L1, G1, Δt, n_iters, σ, k, K; mask=mask1)
    L1 = guided_aeed(L1, G1, Δt, n_iters, σ, k, K; mask=mask1)

    img_out = L0 + Images.imresize(L1, size(L0))
    img_out
end
