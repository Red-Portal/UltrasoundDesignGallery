
function osrad_compute_basis(g_x, g_y)
    g_mag      = sqrt(g_x*g_x + g_y*g_y)
    e0_x, e0_y = if (g_mag > 1e-5)
        g_x / g_mag, g_y / g_mag
    else
        1.0, 0.0
    end
    e1_x = -e0_y
    e1_y = e0_x
    e0_x, e0_y, e1_x, e1_y
end

function osrad_compute_coefficient(e0_x, e0_y,
                                   e1_x, e1_y,
                                   coeff,
                                   ctang)
    λ0 = coeff
    λ1 = ctang

    Dxx = λ0*e0_x*e0_x + λ1*e1_x*e1_x
    Dxy = λ0*e0_x*e0_y + λ1*e1_x*e1_y
    Dyy = λ0*e0_y*e0_y + λ1*e1_y*e1_y
    Dxx, Dxy, Dyy
end

function osrad(img, Δt, n_iters, ctang, w)
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)
    coeff2  = Array{Float32}(undef, M, N)

    D_xx = Array{Float32}(undef, M, N)
    D_xy = Array{Float32}(undef, M, N)
    D_yy = Array{Float32}(undef, M, N)

    ProgressMeter.@showprogress for t = 1:n_iters
        compute_icov!(img_src, coeff2, w)
        C2_noise = median(coeff2)
        coeff2   = (1 .+ (1 ./ max.(coeff2, 1e-7))) ./ (1 .+ (1 / max.(C2_noise, 1e-7)))

        @inbounds for j = 1:N
            @inbounds for i = 1:M
                xm = max(i-1, 1)
                xp = min(i+1, M)
                ym = max(j-1, 1)
                yp = min(j+1, N)

                g_x = (img[xp, j]  - img[xm, j])/2
                g_y = (img[i,  yp] - img[i,  ym])/2

                e0_x, e0_y, e1_x, e1_y = osrad_compute_basis(g_x, g_y)
                d_xx, d_xy, d_yy       = osrad_compute_coefficient(e0_x, e0_y,
                                                                   e1_x, e1_y,
                                                                   coeff2[i,j],
                                                                   ctang)
                D_xx[i,j] = d_xx
                D_xy[i,j] = d_xy
                D_yy[i,j] = d_yy
            end
        end
        weickert_matrix_diffusion!(img_dst, img_src, Δt, D_xx, D_xy, D_yy)
        @swap!(img_src, img_dst)
    end
    img_dst
end
