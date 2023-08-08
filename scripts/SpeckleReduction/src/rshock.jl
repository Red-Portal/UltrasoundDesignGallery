
function rshock(img, guide, Δt, r, k; σ=1.0, mask=trues(size(img)...))
    M = size(img, 1)
    N = size(img, 2)

    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)

    I_ηη    = zeros(Float32, M, N)
    G_x     = zeros(Float32, M, N)
    G_y     = zeros(Float32, M, N)
    C_w     = zeros(Float32, M, N)
    C_e     = zeros(Float32, M, N)
    C_s     = zeros(Float32, M, N)
    C_n     = zeros(Float32, M, N)
    G_xx    = zeros(Float32, M, N)
    G_yy    = zeros(Float32, M, N)
    G_xy    = zeros(Float32, M, N)
    ∇L_mag  = zeros(Float32, M, N)

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
            I_c = guide[i, j]
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

    for j = 1:N
        for i = 1:M
            if (!mask[i,j])
                continue
            end
            I_c = guide[i,j]
            I_e = fetch_pixel(guide, i+1, j,   mask, I_c)
            I_w = fetch_pixel(guide, i-1, j,   mask, I_c)
            I_s = fetch_pixel(guide, i,   j+1, mask, I_c)
            I_n = fetch_pixel(guide, i,   j-1, mask, I_c)

            C_w[i,j] = pmad_weight(abs(I_w - I_c), k)
            C_e[i,j] = pmad_weight(abs(I_e - I_c), k)
            C_n[i,j] = pmad_weight(abs(I_n - I_c), k)
            C_s[i,j] = pmad_weight(abs(I_s - I_c), k)
        end
    end

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
                ∇L_mag[i,j] = sqrt(Dx.*Dx + Dy.*Dy)
            end
        end

        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                I_c = img_src[i, j]
                I_e = fetch_pixel(img_src, i+1, j,   mask, I_c)
                I_w = fetch_pixel(img_src, i-1, j,   mask, I_c)
                I_s = fetch_pixel(img_src, i,   j+1, mask, I_c)
                I_n = fetch_pixel(img_src, i,   j-1, mask, I_c)

                dshock = (1 - pmad_weight(∇G_mag[i,j], k))*sign(I_ηη_σ[i,j])*∇L_mag[i, j]
                img_dst[i,j] = (img_src[i,j] + Δt*(
                    C_w[i,j]*I_w + C_e[i,j]*I_e + C_n[i,j]*I_n + C_s[i,j]*I_s - r*dshock)) /
                    (1 + Δt*(C_w[i,j] + C_e[i,j] + C_n[i,j] + C_s[i,j]))
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
