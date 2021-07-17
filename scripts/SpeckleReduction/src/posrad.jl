
function posrad(img, Δt, n_iters, σ, ρ;
                gmm=nothing,
                n_classes=0,
                mask=trues(size(img)...),
                fit_freq=5)
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
    D_xx   = Array{Float32}(undef, M, N)
    D_xy   = Array{Float32}(undef, M, N)
    D_yy   = Array{Float32}(undef, M, N)
    G_x    = Array{Float32}(undef, M, N, C)
    G_y    = Array{Float32}(undef, M, N, C)
    G_x_ρ  = Array{Float32}(undef, M, N, C)
    G_y_ρ  = Array{Float32}(undef, M, N, C)
    J_xx_ρ = Array{Float32}(undef, M, N, C)
    J_xy_ρ = Array{Float32}(undef, M, N, C)
    J_yy_ρ = Array{Float32}(undef, M, N, C)

    k_ρ = floor(Int, max(ρ, 1)*6/2)*2 + 1
    k_σ = floor(Int, max(σ, 1)*6/2)*2 + 1

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

            GaussianMixtures.em!(gmm, pixels; nIter=16, varfloor=1e-7)
            idx    = sortperm(gmm.μ[:,1]; rev=true)
            gmm.μ  = gmm.μ[idx,:]
            gmm.Σ  = gmm.Σ[idx,:]
            gmm.w  = gmm.w[idx]

            # m = MixtureModel(Normal.(gmm.μ[:,1], sqrt.(gmm.Σ[:,1])), gmm.w[:,1])
            # display(Plots.plot(x-> pdf(m, x), xlims=[-5, 1]))
            # display(Plots.density!(pixels, normed=true))
            #return
            
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

                D_xx[i,j] = λ1.*v1x_k.*v1x_k + λ2.*v2x_k.*v2x_k
                D_xy[i,j] = λ1.*v1x_k.*v1y_k + λ2.*v2x_k.*v2y_k
                D_yy[i,j] = λ1.*v1y_k.*v1y_k + λ2.*v2y_k.*v2y_k
            end
        end

        weickert_matrix_diffusion!(img_dst, img_src, Δt, D_xx, D_xy, D_yy)
        @swap!(img_src, img_dst)
    end
    img_dst
end
