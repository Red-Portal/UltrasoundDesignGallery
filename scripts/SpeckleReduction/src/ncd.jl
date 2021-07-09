
function ncd(img, dt, n_iters, ρ, α, β, s)
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)

    a      = Array{Float32}(undef, M, N)
    b      = Array{Float32}(undef, M, N)
    c      = Array{Float32}(undef, M, N)
    J_xx   = Array{Float32}(undef, M, N)
    J_xy   = Array{Float32}(undef, M, N)
    J_yy   = Array{Float32}(undef, M, N)
    J_xx_σ = Array{Float32}(undef, M, N)
    J_xy_σ = Array{Float32}(undef, M, N)
    J_yy_σ = Array{Float32}(undef, M, N)

    #kernel_x = Images.OffsetArray([3 10  3;  0 0   0; -3 -10 -3]/32.0, -1:1, -1:1)
    #kernel_y = Images.OffsetArray([3  0 -3; 10 0 -10;  3   0 -3]/32.0, -1:1, -1:1)

    k_ρ = floor(Int, ρ*6/2)*2 + 1
    ProgressMeter.@showprogress for t = 1:n_iters
        for j = 1:N
            for i = 1:M
                I_xp = img_src[min(i+1, M), j]
                I_xm = img_src[max(i-1, 1), j]
                I_yp = img_src[i,           min(j+1, N)]
                I_ym = img_src[i,           max(j-1, 1)]

                g_x = (I_xp - I_xm)/2
                g_y = (I_yp - I_ym)/2

                J_xx[i,j] = g_x*g_x
                J_xy[i,j] = g_x*g_y
                J_yy[i,j] = g_y*g_y
            end
        end

        ImageFiltering.imfilter!(J_xx_σ, J_xx, 
                                 ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
                                 "replicate",
                                 ImageFiltering.Algorithm.FIR())
        ImageFiltering.imfilter!(J_xy_σ, J_xy,
                                 ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
                                 "replicate",
                                 ImageFiltering.Algorithm.FIR())
        ImageFiltering.imfilter!(J_yy_σ, J_yy,
                                 ImageFiltering.Kernel.gaussian((ρ, ρ), (k_ρ, k_ρ)),
                                 "replicate",
                                 ImageFiltering.Algorithm.FIR())

        for j = 1:N
            for i = 1:M
                v1x, v1y, v2x, v2y, λ1, λ2 = eigenbasis_2d(J_xx_σ[i,j],
                                                           J_xy_σ[i,j],
                                                           J_yy_σ[i,j])

                δ  = 1 - β*abs(img[i,j] - img_src[i,j])
                κ  = (λ1 - λ2).^2
                λ1 = α*δ
                λ2 = if κ < s*s
                    α*δ*(1 - κ/(s*s))
                else
                    0
                end

                Dxx = λ1*v1x*v1x + λ2*v2x*v2x
                Dxy = λ1*v1x*v1y + λ2*v2x*v2y
                Dyy = λ1*v1y*v1y + λ2*v2y*v2y

                a[i,j] = Dxx;
                b[i,j] = Dxy;
                c[i,j] = Dyy;
            end
        end

        for j = 1:N
            for i = 1:M
                nx = max(i - 1, 1)
                px = min(i + 1, M)
                ny = max(j - 1, 1)
                py = min(j + 1, N)

                A1 = (1/4)*(b[nx, j ] - b[i, py])
                A2 = (1/2)*(c[i,  py] + c[i, j ])
                A3 = (1/4)*(b[px, j ] + b[i, py])
                A4 = (1/2)*(a[nx, j ] + a[i, j ])
                A6 = (1/2)*(a[px, j ] + a[i, j ])
                A7 = (1/4)*(b[nx, j ] + b[i, ny])
                A8 = (1/2)*(c[i,  ny] + c[i, j ])
                A9 = (1/4)*(b[px, j ] - b[i, ny])

                img_dst[i,j] = (img_src[i,j] + dt*(
                    A1*(img_src[nx, py]) + 
                    A2*(img_src[i,  py]) + 
                    A3*(img_src[px, py]) + 
                    A4*(img_src[nx, j])  + 
                    A6*(img_src[px, j])  + 
                    A7*(img_src[nx, ny]) + 
                    A8*(img_src[i,  ny]) + 
                    A9*(img_src[px, ny])))  /
                    (1 + dt*(A1 + A2 + A3 + A4 + A6 + A7 + A8 + A9))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end
