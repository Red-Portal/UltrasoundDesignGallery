
function cshock(img, Δt, n_iters, r, λ_tilde, a, θ;
                mask=trues(size(img)...))
#=
    Complex Shock Filter

    "Image Enhancement and Denoising by Complex Diffusion Processes"
    Guy Gilboa, Nir Sochen, Yehoshua Y. Zeevi
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 2004

    r:       Magnitude of complex diffusion in gradient direction
    λ_tilde: Amount of diffusion in level-set direction
    a:       Slope of arctan
    theta:   Phase angle of complex part
=##
    M = size(img, 1)
    N = size(img, 2)

    λ       = r*exp(im*θ)
    img_src = Complex{Float32}.(img)
    img_dst = zeros(Complex{Float32}, M, N)

    minmod(a, b) = if real(a)*real(b) > 0
        sign(real(a))*min(abs(a), abs(b))
    else
        0
    end

    #ProgressMeter.@showprogress
    for t = 1:n_iters
        for j = 1:N
            for i = 1:M
                if !mask[i,j]
                    continue
                end
                u      = img_src[i,j]
                u_x⁻   = fetch_pixel(img_src, i-1,   j, mask, u)
                u_x⁺   = fetch_pixel(img_src, i+1,   j, mask, u)
                u_y⁻   = fetch_pixel(img_src,   i, j-1, mask, u)
                u_y⁺   = fetch_pixel(img_src,   i, j+1, mask, u)
                u_x⁺y⁺ = fetch_pixel(img_src, i+1, j+1, mask, u)
                u_x⁺y⁻ = fetch_pixel(img_src, i+1, j-1, mask, u)
                u_x⁻y⁺ = fetch_pixel(img_src, i-1, j+1, mask, u)
                u_x⁻y⁻ = fetch_pixel(img_src, i-1, j-1, mask, u)

                Dx_u  = (u_x⁺ - u_x⁻)/2
                Dy_u  = (u_y⁺ - u_y⁻)/2
                Dxx_u = u_x⁺ + u_x⁻ - 2*u
                Dyy_u = u_y⁺ + u_y⁻ - 2*u
                Dxy_u = ((u_x⁺y⁺ - u_x⁻y⁺)/2 - (u_x⁺y⁻ - u_x⁻y⁻)/2)/2

                Dηη_nom = Dxx_u*Dx_u^2 + 2*Dxy_u*Dx_u*Dy_u + Dyy_u*Dy_u^2
                Dηη_den = Dx_u^2 + Dy_u^2 .+ 1e-5
                Dηη_u   = Dηη_nom / Dηη_den

                Dξξ_nom = Dxx_u*Dx_u^2 - 2*Dxy_u*Dx_u*Dy_u + Dyy_u*Dy_u^2
                Dξξ_den = Dηη_den
                Dξξ_u   = Dξξ_nom / Dξξ_den

                D_u_norm = sqrt(minmod(u_x⁺ - u, u - u_x⁻)^2 + minmod(u_y⁺ - u, u - u_y⁻)^2)
               
                img_dst[i,j] = u + Δt*(-2/π*atan(a/θ*imag(u))*D_u_norm + λ*Dηη_u + λ_tilde*Dξξ_u)
            end
        end
        @swap!(img_src, img_dst)
    end
    real.(img_dst)
end
