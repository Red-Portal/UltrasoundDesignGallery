
function admss(img, Δt, n_iters, memory, σ, ρ;
               tissue_selective=true,
               gmm=nothing,
               n_classes=0,
               mask=ones(size(img)...),
               fit_freq=5)
#=
    Anisotropic Diffusion Filter with Memory Based on Speckle Statistics 
    
    "Anisotropic Diffusion Filter With Memory Based on Speckle Statistics for Ultrasound Images"
    Gabriel Ramos-Llordén et al., 
    IEEE Transactions on Image Processing, 2015
=##
    if !isnothing(gmm)
        n_classes = length(gmm.μ)
        fit_freq  = 1
    end

    M = size(img, 1)
    N = size(img, 2)
    C = n_classes

    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)

    img_σ  = Array{Float32}(undef, M, N)
    L_xx   = Array{Float32}(undef, M, N)
    L_xy   = Array{Float32}(undef, M, N)
    L_yy   = Array{Float32}(undef, M, N)
    γ1     = Array{Float32}(undef, M, N)
    γ2     = Array{Float32}(undef, M, N)
    G_x    = Array{Float32}(undef, M, N, C)
    G_y    = Array{Float32}(undef, M, N, C)
    G_x_ρ  = Array{Float32}(undef, M, N, C)
    G_y_ρ  = Array{Float32}(undef, M, N, C)
    J_xx_ρ = Array{Float32}(undef, M, N, C)
    J_xy_ρ = Array{Float32}(undef, M, N, C)
    J_yy_ρ = Array{Float32}(undef, M, N, C)

    k_ρ = floor(Int, max(ρ, 1)*3/2)*2 + 1
    k_σ = floor(Int, max(σ, 1)*3/2)*2 + 1

    gradient_kernel_x = Images.OffsetArray([3 10  3;  0 0   0; -3 -10 -3]/32.0, -1:1, -1:1)
    gradient_kernel_y = Images.OffsetArray([3  0 -3; 10 0 -10;  3   0 -3]/32.0, -1:1, -1:1)
    smooth_σ_kernel   = ImageFiltering.Kernel.gaussian((σ, σ),    (k_σ, k_σ   ))
    smooth_ρ_kernel   = ImageFiltering.Kernel.gaussian((ρ, ρ, 1), (k_ρ, k_ρ, 1))
    filter_type       = ImageFiltering.Algorithm.FIR()
    border_type       = "replicate"

    ProgressMeter.@showprogress for t = 1:n_iters
        if (mod(t, fit_freq) == 1)
            pixels = reshape(Float64.(img_src[mask]), :)
            pixels = reshape(pixels, (:,1))

            if (t == 1)
                gmm = GaussianMixtures.GMM(n_classes, pixels, nIter=0)
            end

            GaussianMixtures.em!(gmm, pixels; nIter=32, varfloor=1e-7)
            idx    = sortperm(gmm.μ[:,1]; rev=true)
            gmm.μ  = gmm.μ[idx,:]
            gmm.Σ  = gmm.Σ[idx,:]
            gmm.w  = gmm.w[idx]
        end

        ImageFiltering.imfilter!(img_σ, img_src, smooth_σ_kernel, border_type, filter_type)
        posterior, _ = GaussianMixtures.gmmposterior(gmm, reshape(img_σ, (:,1)))
        posterior    = reshape(posterior, (size(img_src)..., size(posterior, 2)))

        for c = 1:n_classes
            ImageFiltering.imfilter!(view(G_x, :, :, c),
                                     view(posterior, :, :, c),
                                     gradient_kernel_x,
                                     border_type,
                                     filter_type)
            ImageFiltering.imfilter!(view(G_y, :, :, c),
                                     view(posterior, :, :, c),
                                     gradient_kernel_y,
                                     border_type,
                                     filter_type)
        end

        ImageFiltering.imfilter!(G_x_ρ, G_x, smooth_ρ_kernel, border_type, filter_type)
        ImageFiltering.imfilter!(G_y_ρ, G_y, smooth_ρ_kernel, border_type, filter_type)
        structure_tensor!(J_xx_ρ, J_xy_ρ, J_yy_ρ, G_x_ρ, G_y_ρ, smooth_ρ_kernel)

        for j = 1:N
            for i = 1:M
                v1x_k   = 0
                v1y_k   = 0
                v2x_k   = 0
                v2y_k   = 0
                g_v_mag = 0
                λ_k     = -Inf
                for c = 1:C
                    v1x, v1y, v2x, v2y, λ1, _ = eigenbasis_2d(J_xx_ρ[i,j,c],
                                                              J_xy_ρ[i,j,c],
                                                              J_yy_ρ[i,j,c])
                    if (λ1 > λ_k)
                        v1x_k   = v1x
                        v1y_k   = v1y
                        v2x_k   = v2x
                        v2y_k   = v2y
                        λ_k     = λ1
                        g_v_mag = abs(v1x_k*G_x[i,j,c] + v1y_k*G_y[i,j,c])
                    end
                end

                λ1  = 1.0 .- g_v_mag
                λ2  = 1.0 

                ϵ        = eps(Float32)
                p_tissue = posterior[i, j, C]
                β        = (1 - p_tissue) / (p_tissue.^memory + ϵ)

                if (tissue_selective)
                    λ1 *= p_tissue
                    λ2 *= p_tissue
                end

                γ1[i,j] = if (t == 1)
                    λ1
                else
                    1/(1 + β)*(β*γ1[i,j] + λ1)
                end

                γ2[i,j] = if (t == 1)
                    λ2
                else
                    1/(1 + β)*(β*γ2[i,j] + λ2)
                end

                L_xx[i,j] = γ1[i,j].*v1x_k.*v1x_k + γ2[i,j].*v2x_k.*v2x_k
                L_xy[i,j] = γ1[i,j].*v1x_k.*v1y_k + γ2[i,j].*v2x_k.*v2y_k
                L_yy[i,j] = γ1[i,j].*v1y_k.*v1y_k + γ2[i,j].*v2y_k.*v2y_k
            end
        end
        weickert_matrix_diffusion!(img_dst, img_src, Δt, L_xx, L_xy, L_yy)
        @swap!(img_src, img_dst)
    end
    img_dst
end
