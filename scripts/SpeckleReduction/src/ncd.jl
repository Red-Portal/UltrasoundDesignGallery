
function ncd(img, Δt, n_iters, ρ, α, β, s)
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

    k_ρ = floor(Int, max(ρ, 1)*3/2)*2 + 1
    smooth_kernel = ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ))
    border_type   = "replicate"
    filter_type   = ImageFiltering.Algorithm.FIR()
    ProgressMeter.@showprogress for t = 1:n_iters
        @inbounds for j = 1:N
            @inbounds for i = 1:M
                I_xp = img_src[min(i+1, M), j]
                I_xm = img_src[max(i-1, 1), j]
                I_yp = img_src[i,           min(j+1, N)]
                I_ym = img_src[i,           max(j-1, 1)]

                G_x[i,j] = (I_xp - I_xm)/2
                G_y[i,j] = (I_yp - I_ym)/2
            end
        end
        structure_tensor!(J_xx_ρ, J_xy_ρ, J_yy_ρ, G_x, G_y, smooth_kernel,
                          border_type=border_type,
                          filter_type=filter_type)

        @inbounds for j = 1:N
           @inbounds for i = 1:M
                v1x, v1y, v2x, v2y, λ1, λ2 = eigenbasis_2d(J_xx_ρ[i,j],
                                                           J_xy_ρ[i,j],
                                                           J_yy_ρ[i,j])

                δ  = 1 - β*abs(img[i,j] - img_src[i,j])
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
        weickert_matrix_diffusion!(img_dst, img_src, Δt, D_xx, D_xy, D_yy)
        @swap!(img_src, img_dst)
    end
    img_dst
end
