
function minmod(x, y)
    if (x*y >= 0)
        sign(x)*min(abs(x), abs(y))
    else
        0
    end
end

function ncsf(img, Δt, n_iters; mask=trues(size(img)...))
#=
    Nonconservative shock filter scheme

    "Image Enhancement With PDEs and Nonconservative Advection Flow Fields"
    Vincent Jaouen et al., 2019
=##
    M, N = size(img)

    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)
    D_x     = zeros(Float32, M, N)
    D_y     = zeros(Float32, M, N)
    D_mag   = zeros(Float32, M, N)

    γ = 4
    kernel_size = 7
    kernel_half = kernel_size >> 1
    kernel_x = Images.OffsetArray(zeros(kernel_size, kernel_size),
                                  -kernel_half:kernel_half,
                                  -kernel_half:kernel_half)
    kernel_y = Images.OffsetArray(zeros(kernel_size, kernel_size),
                                  -kernel_half:kernel_half,
                                  -kernel_half:kernel_half)
    for idx in CartesianIndices(kernel_x)
        x, y = Tuple(idx)
        r    = sqrt(x*x + y*y + eps(Float32))
        kernel_x[idx] = x / r^γ
        kernel_y[idx] = y / r^γ
    end

    ProgressMeter.@showprogress for t = 1:n_iters
        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                I_c  = img_src[i,j]
                I_x₊ = fetch_pixel(img_src, i+1,   j, mask, I_c)
                I_x₋ = fetch_pixel(img_src, i-1,   j, mask, I_c)
                I_y₊ = fetch_pixel(img_src,   i, j+1, mask, I_c)
                I_y₋ = fetch_pixel(img_src,   i, j-1, mask, I_c)

                D_x[i,j]   = minmod(I_x₊ - I_c, I_c - I_x₋)
                D_y[i,j]   = minmod(I_y₊ - I_c, I_c - I_y₋)
                D_mag[i,j] = sqrt.(D_x[i,j].^2 + D_y[i,j].^2)
            end
        end
        F_x   = ImageFiltering.imfilter(D_mag.^2, kernel_x)
        F_y   = ImageFiltering.imfilter(D_mag.^2, kernel_y)
        F_mag = sqrt.(F_x.*F_x + F_y.*F_y)
        P     = (D_x.*F_x + D_y.*F_y) ./ (D_mag.*F_mag .+ eps(Float32))
        shock = P.*D_mag     

        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                img_dst[i,j] = img_src[i,j] - Δt*shock[i,j]
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end

function ncrsf(img, Δt, n_iters, λ; mask=trues(size(img)...))
#=
    Nonconcervative shock filter scheme

    "Image Enhancement With PDEs and Nonconservative Advection Flow Fields"
    Vincent Jaouen et al., 2019
=##
    M, N = size(img)

    img_src = deepcopy(img)
    img_dst = zeros(Float32, M, N)
    D_x     = zeros(Float32, M, N)
    D_y     = zeros(Float32, M, N)
    D_mag   = zeros(Float32, M, N)

    γ = 2
    kernel_size = 15
    kernel_half = kernel_size >> 1
    kernel_x = Images.OffsetArray(zeros(kernel_size, kernel_size),
                                  -kernel_half:kernel_half,
                                  -kernel_half:kernel_half)
    kernel_y = Images.OffsetArray(zeros(kernel_size, kernel_size),
                                  -kernel_half:kernel_half,
                                  -kernel_half:kernel_half)
    for idx in CartesianIndices(kernel_x)
        x, y = Tuple(idx)
        r    = sqrt(x*x + y*y + eps(Float32))
        kernel_x[idx] = x / r^γ
        kernel_y[idx] = y / r^γ
    end

    ProgressMeter.@showprogress for t = 1:n_iters
        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                I_c  = img_src[i,j]
                I_x₊ = fetch_pixel(img_src, i+1,   j, mask, I_c)
                I_x₋ = fetch_pixel(img_src, i-1,   j, mask, I_c)
                I_y₊ = fetch_pixel(img_src,   i, j+1, mask, I_c)
                I_y₋ = fetch_pixel(img_src,   i, j-1, mask, I_c)

                D_x[i,j]   = minmod(I_x₊ - I_c, I_c - I_x₋)
                D_y[i,j]   = minmod(I_y₊ - I_c, I_c - I_y₋)
                D_mag[i,j] = sqrt.(D_x[i,j].^2 + D_y[i,j].^2)
            end
        end

        F_x   = ImageFiltering.imfilter(D_mag.^2, kernel_x)
        F_y   = ImageFiltering.imfilter(D_mag.^2, kernel_y)
        F_mag = sqrt.(F_x.*F_x + F_y.*F_y)
        P     = (D_x.*F_x + D_y.*F_y) ./ (D_mag.*F_mag .+ eps(Float32))
        shock = P.*D_mag     

        for j = 1:N
            for i = 1:M
                if (!mask[i,j])
                    continue
                end
                img_dst[i,j] = img_src[i,j] + Δt*(-shock[i,j] + λ*(img[i,j] - img_src[i,j]))
            end
        end
        @swap!(img_src, img_dst)
    end
    img_dst
end
