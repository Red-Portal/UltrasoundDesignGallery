
function ncd(img, Δt, n_iters, ρ, α, β, s; mask=trues(size(img)...))
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)

    D_xx   = zeros(Float32, M, N)
    D_xy   = zeros(Float32, M, N)
    D_yy   = zeros(Float32, M, N)
    G_x    = zeros(Float32, M, N)
    G_y    = zeros(Float32, M, N)
    j1     = zeros(Float32, M, N)
    j2     = zeros(Float32, M, N)
    J_xx_ρ = zeros(Float32, M, N)
    J_xy_ρ = zeros(Float32, M, N)
    J_yy_ρ = zeros(Float32, M, N)

    k_ρ = 9
    smooth_kernel = ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ))
    border_type   = "replicate"
    filter_type   = ImageFiltering.Algorithm.FIR()

    #ProgressMeter.@showprogress
    for t = 1:n_iters
        @inbounds for j = 1:N
            @inbounds for i = 1:M
                if !mask[i,j]
                    continue
                end
                I_c  = img_src[i, j]
                I_e  = fetch_pixel(img_src, i+1,   j, mask, I_c)
                I_se = fetch_pixel(img_src, i+1, j+1, mask, I_c)
                I_ne = fetch_pixel(img_src, i+1, j-1, mask, I_c)
                I_w  = fetch_pixel(img_src, i-1,   j, mask, I_c)
                I_s  = fetch_pixel(img_src, i,   j+1, mask, I_c)
                I_sw = fetch_pixel(img_src, i-1, j+1, mask, I_c)
                I_nw = fetch_pixel(img_src, i-1, j-1, mask, I_c)
                I_n  = fetch_pixel(img_src, i,   j-1, mask, I_c)

                G_y[i,j] = (-3*I_nw + -10*I_w + -3*I_sw +  3*I_ne +  10*I_e +  3*I_se)/32
                G_x[i,j] = ( 3*I_nw +  10*I_n +  3*I_ne + -3*I_sw + -10*I_s + -3*I_se)/32
            end
        end
        structure_tensor!(J_xx_ρ, J_xy_ρ, J_yy_ρ, G_x, G_y, smooth_kernel,
                          border_type=border_type,
                          filter_type=filter_type)

        @inbounds for j = 1:N
           @inbounds for i = 1:M
                if !mask[i,j]
                    continue
                end
                v1x, v1y, v2x, v2y, λ1, λ2 = eigenbasis_2d(J_xx_ρ[i,j],
                                                           J_xy_ρ[i,j],
                                                           J_yy_ρ[i,j])


                δ  = 1 - β*abs(img[i,j] - img_src[i,j])/255
                κ  = (λ1 - λ2).^2
                λ1 = if κ < s*s
                    α*δ*(1 - κ/(s*s))
                else
                    0
                end
                λ2 = α*δ

                D_xx[i,j] = λ1*v1x*v1x + λ2*v2x*v2x
                D_xy[i,j] = λ1*v1x*v1y + λ2*v2x*v2y
                D_yy[i,j] = λ1*v1y*v1y + λ2*v2y*v2y
            end
        end
        rotation_invariant_diffusion!(img_dst, img_src, Δt,
                                      D_xx, D_xy, D_yy,
                                      j1, j2;
                                      mask=mask)
        # weickert_matrix_diffusion!(img_dst, img_src, Δt,
        #                            D_xx, D_xy, D_yy;
        #                            mask=mask)
        @swap!(img_src, img_dst)
    end
    img_dst
end
