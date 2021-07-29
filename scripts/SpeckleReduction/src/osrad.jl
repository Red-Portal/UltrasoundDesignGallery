
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

function osrad(img, Δt, n_iters, ctang, w;
               mask=trues(size(img)...), robust::Bool=true)
#=
    "Oriented Speckle Reducing Anisotropic Diffusion"
    Karl Krissian, Carl-Fredrik Westin, Ron Kikinis, Kirby G. Vosburgh,
    IEEE Transactions on Image Processing, 2007
=##
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)
    coeff2  = zeros(Float32, M, N)

    D_xx = zeros(Float32, M, N)
    D_xy = zeros(Float32, M, N)
    D_yy = zeros(Float32, M, N)

    ProgressMeter.@showprogress for t = 1:n_iters
        compute_icov!(img_src, coeff2, w)
        noise2 = median(coeff2[mask])

        @inbounds for j = 1:N
            @inbounds for i = 1:M
                if !mask[i,j]
                    continue
                end
                coeff2[i,j] = if robust
                    #= 
                        "A robust detail preserving anisotropic diffusion for 
                         speckle reduction in ultrasound images"
                        Xiaoming Liu, Jun Liu, Xin Xu, Lei Chun, Jinshan Tang, Youping Deng 
                        BMC Genomics, 2011
                    =## 
                    R = (coeff2[i,j] .- noise2) ./
                        max.(noise2*(1 .+ coeff2[i,j]), 1e-7)
                    0.5*tukey_biweight(R, 1.0)
                else
                    (1 + (1 / max(coeff2[i,j], 1e-7))) /
                        (1 + (1 / max(noise2, 1e-7)))
                end
            end
        end

        @inbounds for j = 1:N
            @inbounds for i = 1:M
                if !mask[i,j] 
                    continue
                end
                u_c  = img_src[i,  j]
                u_xp = fetch_pixel(img_src, i+1,   j, mask, u_c)
                u_xm = fetch_pixel(img_src, i-1,   j, mask, u_c)
                u_yp = fetch_pixel(img_src,   i, j+1, mask, u_c)
                u_ym = fetch_pixel(img_src,   i, j-1, mask, u_c)

                g_x = (u_xp - u_xm)/2
                g_y = (u_yp - u_ym)/2

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
        weickert_matrix_diffusion!(img_dst, img_src, Δt, D_xx, D_xy, D_yy; mask=mask)
        @swap!(img_src, img_dst)
    end
    img_dst
end
