
function eed(img, Δt, n_iters, σ::Real, λ::Real; mask=trues(size(img)...))
#=
    Edge-enhancing diffusion
    
    "Theoretical Foundations of Anisotropic Diffusion in Image Processing"
    Weickert, Joachim
    Theoretical Foundations of Computer Vision, 1996.

    "Anisotropic Diffusion in Image Processing"
    Weickert, Joachim
    1998
=##
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)

    D_xx   = zeros(Float32, M, N)
    D_xy   = zeros(Float32, M, N)
    D_yy   = zeros(Float32, M, N)
    G_x_σ  = zeros(Float32, M, N)
    G_y_σ  = zeros(Float32, M, N)

    C   = 3.31488
    k_σ = floor(Int, max(σ, 1)*3/2)*2 + 1
    smooth_σ_kernel = ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ))
    filter_type     = ImageFiltering.Algorithm.FIR()
    border_type     = "replicate"

    ProgressMeter.@showprogress for t = 1:n_iters
        img_src_σ = ImageFiltering.imfilter(img_src, smooth_σ_kernel, border_type, filter_type)

        @inbounds for j = 1:N
            @inbounds for i = 1:M
                if !mask[i,j]
                    continue
                end

                I_c  = img_src[i,j]
                I_xp = fetch_pixel(img_src_σ, i+1,   j, mask, I_c)
                I_xm = fetch_pixel(img_src_σ, i-1,   j, mask, I_c)
                I_yp = fetch_pixel(img_src_σ,   i, j+1, mask, I_c)
                I_ym = fetch_pixel(img_src_σ,   i, j-1, mask, I_c)

                G_x_σ[i,j] = (I_xp - I_xm)/2
                G_y_σ[i,j] = (I_yp - I_ym)/2
            end
        end


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

                s² = abs(G_x_σ[i,j].*v1x  + G_y_σ[i,j].*v2x)
                λ1 = if s² > 1e-5
                    1 - exp(-C / (s² / λ^2).^4)
                else
                    1
                end
                λ2 = 1

                D_xx[i,j] = λ1*v1x*v1x + λ2*v2x*v2x
                D_xy[i,j] = λ1*v1x*v1y + λ2*v2x*v2y
                D_yy[i,j] = λ1*v1y*v1y + λ2*v2y*v2y
            end
        end
        weickert_matrix_diffusion!(img_dst, img_src, Δt, D_xx, D_xy, D_yy; mask=mask)
        @swap!(img_src, img_dst)
    end
    img_dst
end


function ieed(img, Δt, n_iters, ρ::Real,
              m1::Real,  m2::Real,
              Cm1::Real, Cm2::Real,
              k1::Real,  k2::Real,
              ; mask=trues(size(img)...))
#=
    Improved Edge-enhancing diffusion
    
=##
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)

    D_xx = zeros(Float32, M, N)
    D_xy = zeros(Float32, M, N)
    D_yy = zeros(Float32, M, N)
    G_x  = zeros(Float32, M, N)
    G_y  = zeros(Float32, M, N)

    k_ρ = floor(Int, max(ρ, 1)*3/2)*2 + 1
    smooth_σ_kernel = ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ))
    filter_type     = ImageFiltering.Algorithm.FIR()
    border_type     = "replicate"

    ProgressMeter.@showprogress for t = 1:n_iters
        @inbounds for j = 1:N
            @inbounds for i = 1:M
                if !mask[i,j]
                    continue
                end

                I_c  = img_src[i,j]
                I_xp = fetch_pixel(img_src, i+1,   j, mask, I_c)
                I_xm = fetch_pixel(img_src, i-1,   j, mask, I_c)
                I_yp = fetch_pixel(img_src,   i, j+1, mask, I_c)
                I_ym = fetch_pixel(img_src,   i, j-1, mask, I_c)

                G_x[i,j] = (I_xp - I_xm)/2
                G_y[i,j] = (I_yp - I_ym)/2
            end
        end

        J_xx = G_x.*G_x
        J_xy = G_x.*G_y
        J_yy = G_y.*G_y

        J_xx_ρ = ImageFiltering.imfilter(J_xx, smooth_σ_kernel, border_type, filter_type)
        J_xy_ρ = ImageFiltering.imfilter(J_xy, smooth_σ_kernel, border_type, filter_type)
        J_yy_ρ = ImageFiltering.imfilter(J_yy, smooth_σ_kernel, border_type, filter_type)

        @inbounds for j = 1:N
            @inbounds for i = 1:M
                if !mask[i,j]
                    continue
                end

                v1x, v1y, v2x, v2y, μ1, μ2 = eigenbasis_2d(J_xx_ρ[i,j],
                                                           J_xy_ρ[i,j],
                                                           J_yy_ρ[i,j])

                λ1 = 1 - exp(-Cm1 / (μ1/k1)^m1 )
                λ2 = 1 - exp(-Cm2 / (μ2/k2)^m2 )

                D_xx[i,j]  = λ1*v1x*v1x + λ2*v2x*v2x
                D_xy[i,j]  = λ1*v1x*v1y + λ2*v2x*v2y
                D_yy[i,j]  = λ1*v1y*v1y + λ2*v2y*v2y
            end
        end
        weickert_matrix_diffusion!(img_dst, img_src, Δt, D_xx, D_xy, D_yy; mask=mask)
        @swap!(img_src, img_dst)
    end
    img_dst
end
