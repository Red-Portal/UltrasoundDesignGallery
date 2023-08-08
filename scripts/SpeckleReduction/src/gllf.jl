
function linesearch(f, x0, x_max, x_min, n_evals)
    x_range = range(x_min, x_max; length=n_evals)
    idx_opt = argmin(f.(x_range))
    x_range[idx_opt]
end

function gllf(img, m, σ, σ_r, K, L, R = maximum(img))
#=
    Gaussian local Laplacian pyramid filter

    Y. Sumiya, T. Otsuka, Y. Maeda, N. Fukushima,
    "Gaussian Fourier Pyramid for Local Laplacian Filter."
    in IEEE Signal Processing Letters, 2022.
=##
    G       = Images.gaussian_pyramid(img, L-1, 2, σ)
    T       = linesearch(0.0, R*2, 0.0, 128) do T
        SpecialFunctions.erfc(π*σ/T*(2*K + 1)) + SpecialFunctions.erfc((T - R)/σ)
    end

    ω       = 2*π*(0:K-1)/T

    G_sin  = map(1:K) do k
        img_sin = sin.(ω[k]*img)
        Images.gaussian_pyramid(img_sin, L-1, 2, σ)
    end

    G_cos  = map(1:K) do k
        img_cos = cos.(ω[k]*img)
        Images.gaussian_pyramid(img_cos, L-1, 2, σ)
    end

    L = map(1:L) do l
        if l == 1
            α = σ_r*sqrt(2*π)/T*exp.(-1/2*(ω*σ_r).^2)

            fourier_recon = mapreduce(+, 1:K) do k
                G_cos_up = Images.imresize(G_cos[k][2], size(G[1]))
                G_sin_up = Images.imresize(G_sin[k][2], size(G[1]))
                α_tilde  = α[k]*2*σ_r^2*ω[k]*m

                α_tilde*(sin.(ω[k]*G[1]).*G_cos_up - cos.(ω[k]*G[1]).*G_sin_up)
            end
            img - Images.imresize(G[2], size(G[1])) + fourier_recon
        elseif l == L
            G[end]
        else
            α       = σ_r*sqrt(2*π)/T*exp.(-1/2*(ω*σ_r).^2)
            Gₗ      = G[l]
            Gₗ₊₁_up = Images.imresize(G[l+1], size(G[l]))
            fourier_recon = mapreduce(+, 1:K) do k
                Cₗ   = G_cos[k][l]
                Sₗ   = G_sin[k][l]

                Cₗ₊₁    = G_cos[k][l+1]
                Sₗ₊₁    = G_sin[k][l+1]
                Cₗ₊₁_up = Images.imresize(Cₗ₊₁, size(G[l]))
                Sₗ₊₁_up = Images.imresize(Sₗ₊₁, size(G[l]))
                sinωₖGₗ = sin.(ω[k]*G[l])
                cosωₖGₗ = cos.(ω[k]*G[l])
                α_tilde = α[k]*2*σ_r^2*ω[k]*m

                α_tilde*(-sinωₖGₗ.*Cₗ + cosωₖGₗ.*Sₗ + sinωₖGₗ.*Cₗ₊₁_up - cosωₖGₗ.*Sₗ₊₁_up)
            end
            Gₗ - Gₗ₊₁_up + fourier_recon
        end
    end
    L
end
