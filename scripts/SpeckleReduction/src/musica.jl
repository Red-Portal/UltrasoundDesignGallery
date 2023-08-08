
function musica(img, p::Array{<:Real}, G::Array{<:Real}, M::Real, σ::Real, L::Int)
#=
    Multiscale Image Contrast Amplification

    P. Vuylsteke and E. Schoeters
    “Multiscale image contrast amplification (MUSICA™),”
    in Proc. SPIE Image Processing, 1994.

    M. Stahl, T. Aach, T. M. Buzug, S. Dippel, and U. Neitzel,
    “Noise-resistant weak-structure enhancement for digital radiography,”
    in Proc. SPIE Medical Imaging, 1999.

    S. Dippel, M. Stahl, R. Wiemker, and T. Blafffert
    "Multiscale Contrast Enhancement for Radiographies: Laplacian Pyramid
     Versus Fast Wavelet Transform,"
    in IEEE Transactions on Medical Imaging, 2002.
=##
    pyr = laplacian_pyramid(img, σ, L)

    r(x, pₗ, Gₗ) = begin
        if abs(x) ≤ M
            Gₗ*x*(1 - abs(x)/M)^pₗ + x
        else
            x
        end
    end

    for l = length(pyr)-1:-1:1
        pyr[l]  = r.(pyr[l], p[l], G[l])
        pyr[l] += Images.imresize(pyr[l+1], size(pyr[l]))
    end
    pyr[1]
end
