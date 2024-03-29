
macro swap!(x,y)
   quote
      local tmp = $(esc(x))
      $(esc(x)) = $(esc(y))
      $(esc(y)) = tmp
    end
end

function eigenbasis_2d(A11, A12, A22)
    tmp = sqrt((A11 - A22).^2 + 4*A12.^2);
    v2x = 2*A12
    v2y = A22 - A11 + tmp;

    mag = sqrt(v2x.^2 + v2y.^2)
    if (mag > eps(Float32))
        v2x /= mag;
        v2y /= mag;
    else
        v2x = 1.0
        v2y = 0.0
    end

    v1x = -v2y
    v1y = v2x

    λ1 = 0.5*(A11 + A22 - tmp)
    λ2 = 0.5*(A11 + A22 + tmp)

    if (abs(λ1) >= abs(λ2))
        v1x, v1y, v2x, v2y, λ1, λ2
    else
        v2x, v2y, v1x, v1y, λ2, λ1
    end 
end

function pmad_weight(x, k)
    r = (x / k)
    1 / (1 + r*r)
end

function tukey_biweight(x, σ)
    if(abs(x) < σ)
        r = x / σ
        c = 1 - r*r
        c*c
    else
        0
    end
end

function lorentzian_error(x, k)
    z = x / k
    exp(-z*z)
end

function huber_minmax(x, σ)
    if(abs(x) < σ)
        1/σ
    else
        sign(x)/x
    end
end

function logcompress(img::Array,
                     max_input_intensity::Real,
                     max_output_intensity::Real,
                     dynamic_range::Real)
    x_max  = max_input_intensity
    x_min  = 10.0.^(-dynamic_range/20)*x_max
    y_max  = max_output_intensity 
    coeff  = y_max / log10(x_max / x_min)

    map(img) do x
        if (x >= x_min)
            coeff*log10(x / x_min)
        else
            0.0
        end
    end
end

function adjust_dynamic_range(img, reject_level, dynamic_range)
    max_intensity = min(reject_level + dynamic_range, 1)
    dynamic_range = max_intensity - reject_level
    img_out       = deepcopy(img)
    for x = 1:size(img, 1)
        for y = 1:size(img, 2)
            u = img[x,y]
            img_out[x,y] = if (u < reject_level)
                0
            elseif (u > max_intensity)
                1
            else
                (u - reject_level) / dynamic_range
            end
        end
    end
    img_out
end


@inline function structure_tensor!(J_xx_ρ::Array,
                                   J_xy_ρ::Array,
                                   J_yy_ρ::Array,
                                   G_x_σ::Array,
                                   G_y_σ::Array,
                                   ρ_kernel;
                                   border_type="replicate",
                                   filter_type=ImageFiltering.Algorithm.FIR())
                                   
    J_xx = G_x_σ.*G_x_σ
    J_xy = G_x_σ.*G_y_σ
    J_yy = G_y_σ.*G_y_σ

    ImageFiltering.imfilter!(J_xx_ρ, J_xx, ρ_kernel, border_type, filter_type)
    ImageFiltering.imfilter!(J_xy_ρ, J_xy, ρ_kernel, border_type, filter_type)
    ImageFiltering.imfilter!(J_yy_ρ, J_yy, ρ_kernel, border_type, filter_type)
end

@inline function weickert_matrix_diffusion!(img_dst::Array,
                                            img_src::Array,
                                            Δt::Real,
                                            D_xx::Array,
                                            D_xy::Array,
                                            D_yy::Array;
                                            mask=trues(size(img_src)...))
    M = size(img_dst, 1)
    N = size(img_dst, 2)
    a = D_xx
    b = D_xy
    c = D_yy
    for j = 1:N
        for i = 1:M
            if !mask[i,j]
                continue
            end

            a_c  = a[i,j]
            a_x₊ = fetch_pixel(a, i+1,   j, mask, a_c)
            a_x₋ = fetch_pixel(a, i-1,   j, mask, a_c)

            b_c  = b[i,j]
            b_x₊ = fetch_pixel(b, i+1,   j, mask, b_c)
            b_x₋ = fetch_pixel(b, i-1,   j, mask, b_c)
            b_y₊ = fetch_pixel(b,   i, j+1, mask, b_c)
            b_y₋ = fetch_pixel(b,   i, j-1, mask, b_c)

            c_c  = c[i,j]
            c_y₊ = fetch_pixel(c,   i, j+1, mask, c_c)
            c_y₋ = fetch_pixel(c,   i, j-1, mask, c_c)

            u_c  = img_src[i,j]
            u1   = fetch_pixel(img_src, i-1, j+1, mask, u_c)
            u2   = fetch_pixel(img_src,   i, j+1, mask, u_c)
            u3   = fetch_pixel(img_src, i+1, j+1, mask, u_c)
            u4   = fetch_pixel(img_src, i-1,   j, mask, u_c)
            u6   = fetch_pixel(img_src, i+1,   j, mask, u_c)
            u7   = fetch_pixel(img_src, i-1, j-1, mask, u_c)
            u8   = fetch_pixel(img_src,   i, j-1, mask, u_c)
            u9   = fetch_pixel(img_src, i+1, j-1, mask, u_c)

            A1 = (1/4)*(b_x₋ - b_y₊)
            A2 = (1/2)*(c_y₊ + c_c )
            A3 = (1/4)*(b_x₊ + b_y₊)
            A4 = (1/2)*(a_x₋ + a_c )
            A6 = (1/2)*(a_x₊ + a_c )
            A7 = (1/4)*(b_x₋ + b_y₋)
            A8 = (1/2)*(c_y₋ + c_c )
            A9 = (1/4)*(b_x₊ - b_y₋)

            img_dst[i,j] = (img_src[i,j] + Δt*(
                A1*u1 + A2*u2 + A3*u3 + A4*u4 + A6*u6 + A7*u7 + A8*u8 + A9*u9))  /
                    (1 + Δt*(A1 + A2 + A3 + A4 + A6 + A7 + A8 + A9))
        end
    end
end

@inline function rotation_invariant_diffusion!(img_dst::Array,
                                               img_src::Array,
                                               Δt::Real,
                                               D_xx::Array,
                                               D_xy::Array,
                                               D_yy::Array,
                                               j1::Array,
                                               j2::Array;
                                               mask=trues(size(img_src)...))
    # J. Weickert, H. Scharr,
    # "A scheme for coherence-enhancing diffusion filtering with
    #  optimized rotation invariance."
    # Journal of Visual Communication and Image Representation, 2001.

    M = size(img_dst, 1)
    N = size(img_dst, 2)

    @inbounds for j = 1:N
        @inbounds for i = 1:M
            if (!mask[i,j])
                continue
            end
            I_n  = img_src[max(i-1, 1), j]
            I_ne = img_src[max(i-1, 1), min(j+1, N)]
            I_nw = img_src[max(i-1, 1), max(j-1, 1)]

            I_s  = img_src[min(i+1, M), j]
            I_se = img_src[min(i+1, M), min(j+1, N)]
            I_sw = img_src[min(i+1, M), max(j-1, 1)]

            I_e  = img_src[i, min(j+1, N)]
            I_w  = img_src[i, max(j-1, 1)]

            G_x = (-3*I_nw +  3*I_ne + -10*I_w  + 10*I_e  +  -3*I_sw +  3*I_se)/32
            G_y = ( 3*I_nw + 10*I_n  +   3*I_ne + -3*I_sw + -10*I_s  + -3*I_se)/32

            j1[i,j] = D_xx[i,j]*G_x + D_xy[i,j]*G_y
            j2[i,j] = D_xy[i,j]*G_x + D_yy[i,j]*G_y
        end
    end

    for j = 1:N
        for i = 1:M
            if (!mask[i,j])
                continue
            end

            j1_ne = j1[max(i-1, 1), min(j+1, N)]
            j1_nw = j1[max(i-1, 1), max(j-1, 1)]

            j1_se = j1[min(i+1, M), min(j+1, N)]
            j1_sw = j1[min(i+1, M), max(j-1, 1)]

            j1_e  = j1[i, min(j+1, N)]
            j1_w  = j1[i, max(j-1, 1)]

            j2_n  = j2[max(i-1, 1), j]
            j2_ne = j2[max(i-1, 1), min(j+1, N)]
            j2_nw = j2[max(i-1, 1), max(j-1, 1)]

            j2_s  = j2[min(i+1, M), j]
            j2_se = j2[min(i+1, M), min(j+1, N)]
            j2_sw = j2[min(i+1, M), max(j-1, 1)]

            ∂xj₁ = (-3*j1_nw +  3*j1_ne + -10*j1_w  + 10*j1_e  +  -3*j1_sw +  3*j1_se)/32
            ∂yj₂ = ( 3*j2_nw + 10*j2_n  +   3*j2_ne + -3*j2_sw + -10*j2_s  + -3*j2_se)/32

            img_dst[i,j] = img_src[i,j] + Δt*(∂xj₁ + ∂yj₂)
        end
    end
end

@inline function fetch_pixel(img, i, j, mask, pad_val)
    i = clamp(i, 1, size(img,1))
    j = clamp(j, 1, size(img,2))
    if (mask[i, j])
        img[i,j]
    else
        pad_val
    end
end

function butterworth(M, N, n_order::Int, cutoff::Real)
    M_offset = M / 2
    N_offset = N / 2
    D0       = cutoff/2
    kernel   = zeros(M, N)
    for idx ∈ CartesianIndices(kernel)
        θ_x         = (idx[1] - M_offset) / M
        θ_y         = (idx[2] - N_offset) / N
        D           = sqrt(θ_x*θ_x + θ_y*θ_y)
        kernel[idx] = 1 / (1 + (D/D0).^(2*n_order))
    end
    kernel
end
