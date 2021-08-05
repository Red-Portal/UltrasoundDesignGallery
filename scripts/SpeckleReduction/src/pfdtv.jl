
function reisz_transform(img, s_min, s_max)
    M, N       = size(img)
    M_padded   = (M >> 1)*2 + 1
    N_padded   = (N >> 1)*2 + 1
    img_padded = zeros(M_padded, N_padded)
    img_padded[1:M,1:N] = img 
    M_half     = M_padded >> 1
    N_half     = N_padded >> 1

    H      = Images.OffsetArray{Complex{Float64}}(
        zeros(size(img_padded)...), -M_half:M_half, -N_half:N_half)
    radius = Images.OffsetArray(
        zeros(size(img_padded)...), -M_half:M_half, -N_half:N_half)
    F      = FFTW.fft(img_padded, (1, 2))

    for x = -M_half:M_half
        for y = -N_half:N_half
            x′ = x / (2*M_half)
            y′ = y / (2*N_half)
            radius[x,y] = sqrt(x′*x′ + y′*y′)
        end
    end

    for x = -M_half:M_half
        for y = -N_half:N_half
            x′ = x / (2*M_half)
            y′ = y / (2*N_half)
            if (radius[x,y] == 0)
                H[x,y] = (1im*x′ - y′) ./ 1.0
            else
                H[x,y] = (1im*x′ - y′) ./ radius[x,y]
            end
        end
    end
    H      = FFTW.ifftshift(parent(H))
    radius = FFTW.ifftshift(parent(radius))

    ΣPC = zeros(size(img_padded)...)
    _, _, _, Ts = ImagePhaseCongruency.phasecongmono(
        img;
        nscale=5,
        minwavelength=5,
        mult=2.1,
        sigmaonf=0.55,
        k=3,
        cutoff=0.5,
        g=10,
        deviationgain=1.5,
        noisemethod=-1)

    for s = s_min:s_max
        xishu  = sqrt.(π*(4 .^(2.58))*(s.^4.16) / SpecialFunctions.gamma(4.16))
        cauchy = xishu*(radius.^1.58).*exp.(-s*radius)
        bF     = F.*cauchy
        f      = real.(FFTW.ifft(bF, (1, 2)))
        h      = FFTW.ifft(bF.*H, (1, 2))

        h1   = real.(h)
        h2   = imag.(h)
        odd  = sqrt.(h1.^2 + h2.^2)
        even = sqrt.(f.^2)
        PC   = (abs.(odd)-abs.(even).-Ts)./(sqrt.(f.^2 + h1.^2 + h2.^2).+0.0001);
        PC   = max.(PC, 0)
        ΣPC += PC
    end
    ΣPC = clamp.(ΣPC, 0, 1)
    ΣPC[1:M,1:N]
end

function binomial(α, l)
    SpecialFunctions.gamma(α + 1) /
        (SpecialFunctions.gamma(l + 1)*SpecialFunctions.gamma(α - l + 1))
end

function fractional_difference(u, α, mask)
    M, N = size(u)
    ∇u_x = zeros(Float32, M, N)
    ∇u_y = zeros(Float32, M, N)
    for j = 1:N
        for i = 1:M
            if !mask[i,j]
                continue
            end
            α_ij      = α[i,j]
            ∇u_x[i,j] = 0
            ∇u_y[i,j] = 0
            for l = 0:j-1
                if α_ij < l
                    continue
                end
                ∇u_x[i,j] += (-1)^(l)*binomial(α_ij,l)*u[i,j-l]
            end
            for l = 0:i-1
                if α_ij < l
                    continue
                end
                ∇u_y[i,j] += (-1)^(l)*binomial(α_ij,l)*u[i-l,j]
            end
        end
    end
    ∇u_x, ∇u_y
end

function FAD(∇u_x, ∇u_y, ∇u_mag, α, PA, k1, mask)
    M, N  = size(∇u_x)
    fad_x = zeros(Float32, M, N)
    fad_y = zeros(Float32, M, N)
    k1²   = k1*k1

    for j = 1:N
        for i = 1:M
            if !mask[i,j]
                continue
            end
            α_ij       = α[i,j]
            fad_x[i,j] = 0
            fad_y[i,j] = 0
            
            for l = 0:N-j
                if α_ij < l
                    continue
                end
                coef = (-1)^(l)*binomial(α_ij,l)
                fad_x[i,j] += coef*(
                    k1²*∇u_x[i,j+l] /
                        (k1² + ∇u_mag[i,j+l].^2*(1 + 254*PA[i,j+l]).^2))
            end

            for l = 0:M-i
                if !mask[i,j]
                    continue
                end
                if α_ij < l
                    continue
                end
                coef = (-1)^(l)*binomial(α_ij,l)
                fad_y[i,j] += coef*(
                    k1²*∇u_y[i+l,j] /
                        (k1² + ∇u_mag[i+l,j].^2*(1 + 254*PA[i+l,j]).^2))
            end
        end
    end
    fad_x, fad_y
end

function FTV(∇u_x, ∇u_y, α, mask)
    M, N  = size(∇u_x)
    ftv_x = zeros(Float32, M, N)
    ftv_y = zeros(Float32, M, N)
    ϵ     = 0.0001
    for j = 1:N
        for i = 1:M
            if !mask[i,j]
                continue
            end
            α_ij       = α[i,j]
            ftv_x[i,j] = 0
            ftv_y[i,j] = 0
            
            for l = 0:N-j
                if α_ij < l
                    continue
                end
                coef = (-1)^(l)*binomial(α_ij,l)
                ftv_x[i,j] += coef*(
                    ∇u_x[i,j+l] /
                        sqrt(∇u_x[i,j+l].^2 + ∇u_y[i,j+l].^2 + ϵ))
            end

            for l = 0:M-i
                if α_ij < l
                    continue
                end
                coef = (-1)^(l)*binomial(α_ij,l)
                ftv_y[i,j] += coef*(
                    ∇u_y[i+l,j] /
                        sqrt(∇u_x[i+l,j].^2 + ∇u_y[i+l,j].^2 + ϵ))
            end
        end
    end
    ftv_x, ftv_y
end

function pfdtv(img, n_iters, s_min, s_max; mask=trues(size(img)...))
#=
    Phase Asymmetry Ultrasound Despeckling with 
    Fractional Anisotropic Diffusion and Total Variation
    
    Kunqiang Mei, Bin Hu, Baowei Fei, Binjie Qin
    IEEE Transactions on Image Processing, 2019
=##
    img_base = deepcopy(img)
    M, N     = size(img_base)
    k0       = 20
    k1       = k0*exp(-0.05*(n_iters - 1))
    λ        = 0.01
    Δt       = 0.15

    ProgressMeter.@showprogress for t = 1:n_iters
        PA           = reisz_transform(img, s_min, s_max)

        α            = 1 .+ log2.(1 .+ PA.*PA)
        φ            = (PA .- 1).^2
        γ            = PA.*(2 .- PA)
        ∇u_x, ∇u_y   = fractional_difference(img, α, mask)
        ∇u_mag       = sqrt.(∇u_x.*∇u_x + ∇u_y.*∇u_y)
        fad_x, fad_y = FAD(∇u_x, ∇u_y, ∇u_mag, α, PA, k1, mask) 
        ftv_x, ftv_y = FTV(∇u_x, ∇u_y, α, mask)

        img = img - Δt*(
            φ.*(fad_x + fad_y)
            + γ.*(ftv_x + ftv_y)
            + λ*(img - img_base))
    end
    img
end
