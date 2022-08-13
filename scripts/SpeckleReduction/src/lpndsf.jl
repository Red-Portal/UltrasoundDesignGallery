
function pmad_coef(x, k)
    r = (x / k)
    1 / (1 + r*r)
end

function pmad_filtered(img, Δt, mse_threshold, k; mask=trues(size(img)...))
    M = size(img, 1)
    N = size(img, 2)

    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)
    G_x     = zeros(Float32, M, N)
    G_y     = zeros(Float32, M, N)

    σ   = 1.0
    k_σ = 7
    smooth_kernel = ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ))
    filter_type   = ImageFiltering.Algorithm.FIR()
    border_type   = "replicate"

    prog = ProgressMeter.ProgressThresh(mse_threshold, "MSE of ΔI:")
    for t = 1:100
        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                nx = max(i - 1, 1)
                px = min(i + 1, M)
                ny = max(j - 1, 1)
                py = min(j + 1, N)

                I_c = img_src[i,j]
                px  = fetch_pixel(img_src, i+1, j,   mask, I_c)
                nx  = fetch_pixel(img_src, i-1, j,   mask, I_c)
                py  = fetch_pixel(img_src, i,   j+1, mask, I_c)
                ny  = fetch_pixel(img_src, i,   j-1, mask, I_c)

                G_x[i,j] = (px - nx)/2
                G_y[i,j] = (py - ny)/2
            end
        end
        G_x_σ    = ImageFiltering.imfilter(G_x, smooth_kernel, border_type, filter_type)
        G_y_σ    = ImageFiltering.imfilter(G_y, smooth_kernel, border_type, filter_type)
        G_mag    = sqrt.(G_x_σ.*G_x_σ + G_y_σ.*G_y_σ)
        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                I_c = img_src[i,j]
                I_e = fetch_pixel(img_src, i+1, j,   mask, I_c)
                I_w = fetch_pixel(img_src, i-1, j,   mask, I_c)
                I_s = fetch_pixel(img_src, i,   j+1, mask, I_c)
                I_n = fetch_pixel(img_src, i,   j-1, mask, I_c)

                C_w = pmad_coef(G_mag[i, j], k)
                C_e = pmad_coef(G_mag[i, j], k)
                C_n = pmad_coef(G_mag[i, j], k)
                C_s = pmad_coef(G_mag[i, j], k)

                img_dst[i,j] = (img_src[i,j] + Δt*(C_w*I_w + C_e*I_e + C_n*I_n + C_s*I_s)) /
                    (1 + Δt*(C_w + C_e + C_n + C_s))
            end
        end
        ΔI  = (img_src[mask] - img_dst[mask])
        mse = mean(ΔI.*ΔI)
        ProgressMeter.update!(prog, mse)
        if (mse < mse_threshold)
            break
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end

function minmod(x, y)
    if (x*y > 0)
        sign(x)*min(abs(x), abs(y))
    else
        0
    end
end

function pmad_shock(img, guide, Δt, mse_threshold, r, k; mask=trues(size(img)...))
    M = size(img, 1)
    N = size(img, 2)

    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)

    I_ηη    = zeros(Float32, M, N)
    G_x     = zeros(Float32, M, N)
    G_y     = zeros(Float32, M, N)
    G_w     = zeros(Float32, M, N)
    G_e     = zeros(Float32, M, N)
    G_s     = zeros(Float32, M, N)
    G_n     = zeros(Float32, M, N)
    G_xx    = zeros(Float32, M, N)
    G_yy    = zeros(Float32, M, N)
    G_xy    = zeros(Float32, M, N)
    G_mag   = zeros(Float32, M, N)

    σ   = 1.0
    k_σ = 7
    smooth_kernel = ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ))
    filter_type   = ImageFiltering.Algorithm.FIR()
    border_type   = "replicate"

    prog = ProgressMeter.ProgressThresh(mse_threshold, "MSE of ΔI:")

    for j = 1:N
        for i = 1:M
            if (!mask[i,j])
                continue
            end
            I_c = fetch_pixel(guide, i,   j,   mask, 0.0)
            px  = fetch_pixel(guide, i+1, j,   mask, I_c)
            nx  = fetch_pixel(guide, i-1, j,   mask, I_c)
            py  = fetch_pixel(guide, i,   j+1, mask, I_c)
            ny  = fetch_pixel(guide, i,   j-1, mask, I_c)

            G_x[i,j]  = 0.5*(py - ny)
            G_y[i,j]  = 0.5*(px - nx)
            G_xx[i,j] = py + ny - 2*I_c
            G_yy[i,j] = px + nx - 2*I_c
        end
    end

    for j = 1:N
        for i = 1:M
            if (!mask[i,j])
                continue
            end
            px = fetch_pixel(G_x, i+1, j, mask, G_x[i,j])
            nx = fetch_pixel(G_x, i-1, j, mask, G_x[i,j])
            G_xy[i,j] = 0.5*(px - nx)
        end
    end
    for i = 1:M
        for j = 1:N
            if (!mask[i,j])
                continue
            end
            I_ηη[i,j] = G_xx[i,j]*(G_x[i,j]*G_x[i,j]) +
                2*G_xy[i,j]*G_x[i,j]*G_y[i,j] +
                G_yy[i,j]*(G_y[i,j]*G_y[i,j])
        end
    end

    I_ηη_σ = ImageFiltering.imfilter(I_ηη, smooth_kernel, border_type, filter_type)
    ∇G_mag = sqrt.(G_x.*G_x + G_y.*G_y)

    for t = 1:100
        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                nx = max(i - 1, 1)
                px = min(i + 1, M)
                ny = max(j - 1, 1)
                py = min(j + 1, N)

                Δ₊x = img_src[px,j] - img_src[i,j]
                Δ₋x = img_src[i,j]  - img_src[nx,j]
                Δ₊y = img_src[i,py] - img_src[i,j]
                Δ₋y = img_src[i,j]  - img_src[i,ny]

                Dx = minmod(Δ₊x, Δ₋x)
                Dy = minmod(Δ₊y, Δ₋y)
                G_mag[i,j] = sqrt(Dx.*Dx + Dy.*Dy)
            end
        end

        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                nx = max(i - 1, 1)
                px = min(i + 1, M)
                ny = max(j - 1, 1)
                py = min(j + 1, N)

                I_w = img_src[nx, j]
                I_e = img_src[px, j]
                I_n = img_src[i, ny]
                I_s = img_src[i, py]
                I_c = img_src[i, j ]

                G_w[i,j] = I_w - I_c
                G_e[i,j] = I_e - I_c
                G_n[i,j] = I_n - I_c
                G_s[i,j] = I_s - I_c
            end
        end

        G_w_σ = ImageFiltering.imfilter(G_w, smooth_kernel, border_type, filter_type)
        G_s_σ = ImageFiltering.imfilter(G_s, smooth_kernel, border_type, filter_type)
        G_e_σ = ImageFiltering.imfilter(G_e, smooth_kernel, border_type, filter_type)
        G_n_σ = ImageFiltering.imfilter(G_n, smooth_kernel, border_type, filter_type)

        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                nx = max(i - 1, 1)
                px = min(i + 1, M)
                ny = max(j - 1, 1)
                py = min(j + 1, N)

                I_w = img_src[nx, j]
                I_e = img_src[px, j]
                I_n = img_src[i, ny]
                I_s = img_src[i, py]

                C_w = pmad_coef(abs(G_w_σ[i,j]), k)
                C_e = pmad_coef(abs(G_e_σ[i,j]), k)
                C_n = pmad_coef(abs(G_n_σ[i,j]), k)
                C_s = pmad_coef(abs(G_s_σ[i,j]), k)
                dshock = (1 - pmad_coef(∇G_mag[i,j], k))*sign(I_ηη_σ[i,j])*G_mag[i, j]
                img_dst[i,j] = (img_src[i,j] + Δt*(
                    C_w*I_w + C_e*I_e + C_n*I_n + C_s*I_s - r*dshock)) /
                    (1 + Δt*(C_w + C_e + C_n + C_s))
            end
        end
        ΔI  = (img_src[mask] - img_dst[mask])
        mse = mean(ΔI.*ΔI)
        ProgressMeter.update!(prog, mse)
        if (mse < mse_threshold)
            break
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end


function lpndsf(img, Δt, mse_threshold, k, r; mask=trues(size(img)...))
#=
    Laplacian Pyramid-based Nonlinear Diffusion and Shock Filter

    "Multiscale Nonlinear Diffusion and Shock Filter for Ultrasound Image Enhancement"
    Fan Zhang et al., CVPR 2006
=##
    G_pyramid = Images.gaussian_pyramid(img, 4, 2, 3)
    L_pyramid = deepcopy(G_pyramid)

    for i = 1:3
        L_pyramid[i] = G_pyramid[i] - Images.imresize(G_pyramid[i+1], size(G_pyramid[i])...)
    end
    L_pyramid[4] = G_pyramid[4]

    mask_resize = Images.imresize(mask, size(G_pyramid[1])) .> 0.0
    G_denoised  = pmad_filtered(G_pyramid[1], Δt, mse_threshold, k[1]; mask=mask_resize)
    L_pyramid[1] = pmad_shock(L_pyramid[1], G_denoised, Δt, mse_threshold, r[1], k[1]; mask=mask_resize)

    mask_resize  = Images.imresize(mask, size(G_pyramid[2])) .> 0.0
    G_denoised   = pmad_filtered(G_pyramid[2], Δt, mse_threshold, k[2]; mask=mask_resize)
    L_pyramid[2] = pmad_shock(L_pyramid[2], G_denoised, Δt, mse_threshold, r[2], k[2]; mask=mask_resize)

    mask_resize  = Images.imresize(mask, size(G_pyramid[3])) .> 0.0
    G_denoised   = pmad_filtered(G_pyramid[3], Δt, mse_threshold, k[3]; mask=mask_resize)
    L_pyramid[3] = pmad_shock(L_pyramid[3], G_denoised, Δt, mse_threshold, r[3], k[3]; mask=mask_resize)

    for i = 3:-1:1
        L_pyramid[i] += Images.imresize(L_pyramid[i+1], size(L_pyramid[i])...)
    end
    L_pyramid[1]
end
