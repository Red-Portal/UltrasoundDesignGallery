
function apm(img, Δt, n_iters, σ, k, K;
             mask=trues(size(img)...))
#=
    Adaptive Perona–Malik Model Based on the Variable Exponent 
    for Image Denoising
   
    Zhichang Guo, Jiebao Sun, Dazhi Zhang, Boying Wu
    IEEE Transactions on Image Processing, 2012
=##
    M = size(img, 1)
    N = size(img, 2)

    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)
    G_mag   = zeros(Float32, M, N)
    G_mag_σ = zeros(Float32, M, N)

    k_σ             = floor(Int, max(σ, 1)*6/2)*2 + 1
    smooth_σ_kernel = ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ))
    filter_type     = ImageFiltering.Algorithm.FIR()
    border_type     = "replicate"

    ProgressMeter.@showprogress for t = 1:n_iters
        @inbounds for j = 1:N
            @inbounds for i = 1:M
                if !mask[i,j]
                    continue
                end

                I_c = img_src[i,  j]
                I_w = fetch_pixel(img_src, i-1,   j, mask, I_c)
                I_e = fetch_pixel(img_src, i+1,   j, mask, I_c)
                I_n = fetch_pixel(img_src,   i, j-1, mask, I_c)
                I_s = fetch_pixel(img_src,   i, j+1, mask, I_c)

                g_x        = (I_w - I_e)/2
                g_y        = (I_n - I_s)/2
                G_mag[i,j] = sqrt(g_x*g_x + g_y*g_y)
            end
        end

        ImageFiltering.imfilter!(G_mag_σ, G_mag, smooth_σ_kernel, border_type, filter_type)

        α = 2 .- (2 ./ (1 .+ k*G_mag_σ.*G_mag_σ))
        display(Plots.histogram(reshape(α[mask], :)))
        C = 1 ./ (1 .+ (G_mag/K).^α)
        #display(Plots.histogram!(reshape(C[mask], :)))

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

                C_c = C[i,j]
                C_w = fetch_pixel(C, i-1,   j, mask, C_c)
                C_e = fetch_pixel(C, i+1,   j, mask, C_c)
                C_n = fetch_pixel(C,   i, j-1, mask, C_c)
                C_s = fetch_pixel(C,   i, j+1, mask, C_c)

                img_dst[i,j] = (I_c +
                    Δt*(C_w*I_w + C_e*I_e + C_n*I_n + C_s*I_s)) /
                    (1 + Δt*(C_w + C_e + C_n + C_s))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end
