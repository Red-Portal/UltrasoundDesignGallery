
function ced(img, Δt, n_iters, σ, ρ, α, β;
             mask=trues(size(img)...))
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)

    D_xx   = Array{Float32}(undef, M, N)
    D_xy   = Array{Float32}(undef, M, N)
    D_yy   = Array{Float32}(undef, M, N)
    G_x    = Array{Float32}(undef, M, N)
    G_y    = Array{Float32}(undef, M, N)
    J_xx_ρ = Array{Float32}(undef, M, N)
    J_xy_ρ = Array{Float32}(undef, M, N)
    J_yy_ρ = Array{Float32}(undef, M, N)
    img_σ  = Array{Float32}(undef, M, N)

    k_ρ = floor(Int, max(ρ, 1)*3/2)*2 + 1
    k_σ = floor(Int, max(σ, 1)*3/2)*2 + 1
    smooth_σ_kernel = ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ))
    smooth_ρ_kernel = ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ))
    border_type     = "replicate"
    filter_type     = ImageFiltering.Algorithm.FIR()

    ProgressMeter.@showprogress for t = 1:n_iters
        ImageFiltering.imfilter!(img_σ, img_src, smooth_σ_kernel, border_type, filter_type)

        @inbounds for j = 1:N
            @inbounds for i = 1:M
                if (!mask[i,j])
                    continue
                end
                I_xp = img_σ[min(i+1, M), j]
                I_xm = img_σ[max(i-1, 1), j]
                I_yp = img_σ[i, min(j+1, N)]
                I_ym = img_σ[i, max(j-1, 1)]

                G_x[i,j] = (I_xp - I_xm)/2
                G_y[i,j] = (I_yp - I_ym)/2
            end
        end
        structure_tensor!(J_xx_ρ, J_xy_ρ, J_yy_ρ, G_x, G_y, smooth_ρ_kernel,
                          border_type=border_type,
                          filter_type=filter_type)

        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                v1x, v1y, v2x, v2y, λ1, λ2 = eigenbasis_2d(J_xx_ρ[i,j],
                                                           J_xy_ρ[i,j],
                                                           J_yy_ρ[i,j])

                κ  = (λ1 - λ2).^2
                λ1 = α
                λ2 = if abs(κ) > eps(Float32)
                    α + (1 - α)*exp(-β ./ κ)
                else
                    α 
                end

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
