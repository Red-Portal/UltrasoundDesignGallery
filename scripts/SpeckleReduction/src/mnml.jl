
function create_window(window_size)
    search_half   = window_size >> 1
    window        = LocalFilters.RectangularBox{2}(window_size, window_size)
    window, search_half
end

function cartesianregion(start::Tuple, stop::Tuple, scale::Int) where N
    return CartesianIndices(map((i,j) -> i:scale:j, start, stop))
end


function mnml(img::Array{T}, search_size, patch_size, n_scales, h;
              mask=trues(size(img)...)) where {T <: Real}
#=
    Multiscale Non-Local Mean


=##
    M, N         = size(img)
    img_med      = ImageFiltering.mapwindow(img, (5, 5)) do w
        median(w)
    end
    img_md       = img - img_med
    shrink_thres = quantile(filter(x -> x > 0, reshape(img_md[mask], :)), 0.90)
    img_shrunk   = zeros(T, size(img)...)
    @inbounds for j = 1:N
        @inbounds for i = 1:M
            if (!mask[i,j])
                continue
            end
            img_shrunk[i,j] = img[i,j] - max(img_md[i,j] - shrink_thres, 0)
        end
    end

    regions       = LocalFilters.cartesianregion(img)
    search_window = LocalFilters.RectangularBox{2}(search_size, search_size)
    search_half   = search_size >> 1
    patch_window  = LocalFilters.RectangularBox{2}(patch_size, patch_size)
    patch_half    = patch_size >> 1

    pad_size = search_half*(n_scales+1) + patch_half 
    img_src  = Images.padarray(img_shrunk, Images.Pad(pad_size, pad_size))
    search_min, search_max = LocalFilters.limits(search_window)
    patch_min, patch_max   = LocalFilters.limits(patch_window)

    n_valid = sum(mask)
    img_dst = deepcopy(img_src)
    @inbounds for n = 1:n_scales
        prog = ProgressMeter.Progress(n_valid)
        search_min_scaled = CartesianIndex(map(idx -> idx*n, Tuple(search_min)))
        search_max_scaled = CartesianIndex(map(idx -> idx*n, Tuple(search_max)))
        @inbounds for i in regions
            ProgressMeter.next!(prog)
            if !mask[i]
                continue
            end

            Σw     = T(0)
            Σwv    = T(0)
            x0_idx = LocalFilters.cartesianregion(i - patch_max, i - patch_min)
            x0     = @view img_src[x0_idx]

            search_region = cartesianregion(Tuple(i-search_max_scaled),
                                            Tuple(i-search_min_scaled),
                                            n)
            @simd for j in search_region
                x_idx = LocalFilters.cartesianregion(j - patch_max, j - patch_min)
                x     = @view img_src[x_idx]
                Δx    = (x - x0)
                w     = exp(-dot(Δx, Δx) / (2*h*h) )
                Σw   += w
                Σwv  += w*img_src[j]
            end
            img_dst[i] = Σwv / Σw
        end
        h /= 2
        img_src = deepcopy(img_dst)
    end
    img_dst[1:M, 1:N]
end
