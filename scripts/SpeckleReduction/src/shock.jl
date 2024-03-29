
function shock(img, dt, ρ, σ, n_iters; mask=trues(size(img)...))
    kernel_x = Images.OffsetArray([3 10  3;  0 0   0; -3 -10 -3]/32.0, -1:1, -1:1)
    kernel_y = Images.OffsetArray([3  0 -3; 10 0 -10;  3   0 -3]/32.0, -1:1, -1:1)

    M = size(img, 1)
    N = size(img, 2)

    k_ρ = floor(Int, ρ*6/2)*2 + 1
    k_σ = floor(Int, σ*6/2)*2 + 1

    ProgressMeter.@showprogress for t = 1:n_iters
        img_σ = ImageFiltering.imfilter(
            img,
            ImageFiltering.Kernel.gaussian((σ, σ), (k_σ, k_σ)),
            "replicate",
            ImageFiltering.Algorithm.FIR())

        v_x = ImageFiltering.imfilter(
            img_σ,
            kernel_x,
            "replicate",
            ImageFiltering.Algorithm.FIR())
        v_y = ImageFiltering.imfilter(
            img_σ,
            kernel_y,
            "replicate",
            ImageFiltering.Algorithm.FIR())

        v_xx = ImageFiltering.imfilter(
            v_x,
            kernel_x,
            "replicate",
            ImageFiltering.Algorithm.FIR())
        v_xy = ImageFiltering.imfilter(
            v_x,
            kernel_y,
            "replicate",
            ImageFiltering.Algorithm.FIR())
        v_yy = ImageFiltering.imfilter(
            v_y,
            kernel_y,
            "replicate",
            ImageFiltering.Algorithm.FIR())

        u_x = ImageFiltering.imfilter(
            img,
            kernel_x,
            "replicate",
            ImageFiltering.Algorithm.FIR())
        u_y = ImageFiltering.imfilter(
            img,
            kernel_y,
            ImageFiltering.Algorithm.FIR())

        J_xx_σ = ImageFiltering.imfilter(
            u_x .* u_x,
            ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
            "replicate",
            ImageFiltering.Algorithm.FIR())
        J_xy_σ = ImageFiltering.imfilter(
            u_x .* u_y,
            ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
            "replicate",
            ImageFiltering.Algorithm.FIR())
        J_yy_σ = ImageFiltering.imfilter(
            u_y .* u_y,
            ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
            "replicate",
            ImageFiltering.Algorithm.FIR())

        for j = 1:N
            for i = 1:M
                if !mask[i,j]
                    continue
                end

                v1x, v1y, v2x, v2y, λ1, _ = eigenbasis_2d(J_xx_σ[i,j],
                                                          J_xy_σ[i,j],
                                                          J_yy_σ[i,j])
                w_x = v1x
                w_y = v1y
                Δu = -sign(v_xx[i,j]*w_x*w_x + v_xy[i,j]*w_x*w_y + v_yy[i,j]*w_y*w_y)*
                    sqrt(u_x[i,j].^2 + u_y[i,j].^2)

                img[i,j] = img[i,j] + dt*Δu
            end
        end
    end
    img
end
