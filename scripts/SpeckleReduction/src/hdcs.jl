
function hdcs(img, Δt, n_iters,
              σ::Real, ρ::Real, λ_h::Real, λ_eed::Real, λ_ced::Real;
              mask=trues(size(img)...))
#=
    Hybrid Diffusion With Continuous Switch

    "Noise Reduction in Computed Tomography Scans Using 
     3-D Anisotropic Hybrid Diffusion With Continuous Switch"
    AdriËnne M. Mendrik et al.
    IEEE Transactions on Medical Imaging, 2009
=##
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)

    D_xx   = zeros(Float32, M, N)
    D_xy   = zeros(Float32, M, N)
    D_yy   = zeros(Float32, M, N)
    G_x    = zeros(Float32, M, N)
    G_y    = zeros(Float32, M, N)

    α   = 0.001
    C   = 3.31488
    k_ρ = floor(Int, max(ρ, 1)*6/2)*2 + 1
    k_σ = floor(Int, max(σ, 1)*6/2)*2 + 1

    gradient_kernel_x = Images.OffsetArray([3 10  3;  0 0   0; -3 -10 -3]/32.0, -1:1, -1:1)
    gradient_kernel_y = Images.OffsetArray([3  0 -3; 10 0 -10;  3   0 -3]/32.0, -1:1, -1:1)
    smooth_σ_kernel   = ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ))
    smooth_ρ_kernel   = ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ))
    filter_type       = ImageFiltering.Algorithm.FIR()
    border_type       = "replicate"

    ProgressMeter.@showprogress for t = 1:n_iters
        G_x = ImageFiltering.imfilter(img_src, gradient_kernel_x, border_type, filter_type)
        G_y = ImageFiltering.imfilter(img_src, gradient_kernel_y, border_type, filter_type)

        G_x_σ = ImageFiltering.imfilter(G_x, smooth_σ_kernel, border_type, filter_type)
        G_y_σ = ImageFiltering.imfilter(G_y, smooth_σ_kernel, border_type, filter_type)

        J_xx  =  G_x_σ.*G_x_σ
        J_xy  =  G_x_σ.*G_y_σ
        J_yy  =  G_y_σ.*G_y_σ

        J_xx_ρ = ImageFiltering.imfilter(J_xx, smooth_ρ_kernel, border_type, filter_type)
        J_xy_ρ = ImageFiltering.imfilter(J_xy, smooth_ρ_kernel, border_type, filter_type)
        J_yy_ρ = ImageFiltering.imfilter(J_yy, smooth_ρ_kernel, border_type, filter_type)

        fuck = zeros(M,N)

        @inbounds for j = 1:N
            @inbounds for i = 1:M
                if !mask[i,j]
                    continue
                end

                v1x, v1y, v2x, v2y, μ1, μ2 = eigenbasis_2d(
                    J_xx_ρ[i,j], J_xy_ρ[i,j], J_yy_ρ[i,j])

                s²     = G_x_σ[i,j].^2 + G_y_σ[i,j].^2
                λ1_eed = if s² > 1e-5
                    1 - exp(-C / (s² / λ_eed^2).^4)
                else
                    1
                end
                λ2_eed = 1

                κ      = (μ1 - μ2).^2
                λ1_ced = α
                λ2_ced = α + (1 - α)*exp(-log(2)*λ_ced.^2 / κ)

                ξ  = (μ1/(α + μ2) - μ2/α)
                ε  = exp(μ2/(λ_h*λ_h)*((ξ - abs(ξ))/2))
                λ1 = (1 - ε)*λ1_ced + ε*λ1_eed
                λ2 = (1 - ε)*λ2_ced + ε*λ2_eed

                fuck[i,j] = ε

                D_xx[i,j] = λ1*v1x*v1x + λ2*v2x*v2x
                D_xy[i,j] = λ1*v1x*v1y + λ2*v2x*v2y
                D_yy[i,j] = λ1*v1y*v1y + λ2*v2y*v2y
            end
        end
        display(Plots.histogram(reshape(fuck[mask], :), normed=true))
        #ImageView.imshow(fuck)
        #return
        weickert_matrix_diffusion!(img_dst, img_src, Δt, D_xx, D_xy, D_yy; mask=mask)
        @swap!(img_src, img_dst)
    end
    img_dst
end
