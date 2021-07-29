
function rpncd(img, Δt, n_iters, k, θ;
              mask=trues(size(img)...))
#=
    Ramp-Preserving Nonlinear Complex Diffusion

    "Image Enhancement and Denoising by Complex Diffusion Processes"
    Guy Gilboa, Nir Sochen, Yehoshua Y. Zeevi
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 2004
=##
    M = size(img, 1)
    N = size(img, 2)

    img_src = Complex{Float32}.(img)
    img_dst = zeros(Complex{Float32}, M, N)
    coeff   = zeros(Complex{Float32}, M, N)

    ProgressMeter.@showprogress for t = 1:n_iters
        for j = 1:N
            for i = 1:M
                if !mask[i,j]
                    continue
                end
                I          = img_src[i,j]
                coeff[i,j] = exp(im*θ) / (1 + (imag.(I) / (k*θ)).^2)
            end
        end

        #if (t > 3)
        #display(Plots.plot(real.(img_src[256,:])))
        #display(Plots.plot!(real.(coeff[256,:])))
        #return
        #end

        for j = 1:N
            for i = 1:M
                if !mask[i,j]
                    continue
                end
                I_c = img_src[i,j]
                I_w = fetch_pixel(img_src, i-1,   j, mask, I_c)
                I_e = fetch_pixel(img_src, i+1,   j, mask, I_c)
                I_n = fetch_pixel(img_src,   i, j-1, mask, I_c)
                I_s = fetch_pixel(img_src,   i, j+1, mask, I_c)

                C_c = coeff[i,j]
                C_w = fetch_pixel(coeff, i-1,   j, mask, C_c)
                C_e = fetch_pixel(coeff, i+1,   j, mask, C_c)
                C_n = fetch_pixel(coeff,   i, j-1, mask, C_c)
                C_s = fetch_pixel(coeff,   i, j+1, mask, C_c)

                img_dst[i,j] = (I_c +
                   Δt/4*(C_w*(I_w - I_c) + C_e*(I_e - I_c) + C_n*(I_n - I_c) + C_s*(I_s - I_c)))
            end
        end
        @swap!(img_src, img_dst)
    end
    real.(img_dst)
end
