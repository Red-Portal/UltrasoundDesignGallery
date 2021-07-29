
function compute_icov!(img_src, coeff2_dst, w)
    Ex  = ImageFiltering.mapwindow(mean, img_src,          (w, w))
    Ex2 = ImageFiltering.mapwindow(mean, img_src.*img_src, (w, w))
    var = Ex2 - Ex.*Ex
    
    coeff2_dst[:,:] = var ./ max.(Ex.*Ex, 1e-5)
end

function dpad(img, dt, n_iters, w;
              mask=trues(size(img)...),
              robust::Bool=true)
#=
    Detail-Preserivng Anisotropic Diffusion

    "On the estimation of the coefficient of variation 
     for anisotropic diffusion speckle filtering"
    S. Aja-Fernandez, C. Alberola-Lopez
    IEEE Transactions on Image Processing, 2006
=##
    M       = size(img, 1)
    N       = size(img, 2)
    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)
    coeff2  = zeros(Float32, M, N)

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

                c_c  = coeff2[i,j]
                c_xp = fetch_pixel(coeff2, i+1,   j, mask, c_c)
                c_xm = fetch_pixel(coeff2, i-1,   j, mask, c_c)
                c_yp = fetch_pixel(coeff2,   i, j+1, mask, c_c)
                c_ym = fetch_pixel(coeff2,   i, j-1, mask, c_c)

                img_dst[i,j] = (img_src[i,j] + dt*(
                    u_xp*c_xp + u_xm*c_xm + u_yp*c_yp + u_ym*c_ym)) / 
                        (1 + dt*(c_xp + c_xm + c_yp + c_ym))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end

