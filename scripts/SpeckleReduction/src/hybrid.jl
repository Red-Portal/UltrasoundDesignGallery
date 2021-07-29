
function guided(img::Array, I::Array, W::Int, ϵ::Real)
    border_type = "replicate"
    filter_type = ImageFiltering.Algorithm.FIR()

    offset     = div(W, 2)
    avg_filter = Images.OffsetArray(fill(1/(W*W), W, W), -offset:offset, -offset:offset)

    p   = img
    μ   = ImageFiltering.imfilter(I,    avg_filter, border_type, filter_type)
    EI² = ImageFiltering.imfilter(I.*I, avg_filter, border_type, filter_type)
    σ²  = EI² - (μ.*μ)
    Ip  = I.*p
    Ep  = ImageFiltering.imfilter(p,  avg_filter, border_type, filter_type)
    EIp = ImageFiltering.imfilter(Ip, avg_filter, border_type, filter_type)

    a = ImageFiltering.imfilter((Ip .- EIp), avg_filter, border_type, filter_type) ./  (σ² .+ ϵ)
    b = Ep .- (μ.*a)

    μ_a = ImageFiltering.imfilter(a, avg_filter, border_type, filter_type)
    μ_b = ImageFiltering.imfilter(b, avg_filter, border_type, filter_type)

    (μ_a.*I) .+ μ_b
end

function srbf(img::Array, W::Int, σ₁::Real, σ₂::Real)
    σ₁² = σ₁*σ₁
    σ₂² = σ₂*σ₂
    
    f(x::Real, y::Real) = begin
        r = sqrt(x) - sqrt(y)
        exp(-r*r / (2*σ₁²))
    end

    g(I::CartesianIndex{N}) where {N} = begin
        exp(-sum((k) -> I[k]^2, 1:N) / (2*σ₂²))
    end

    LocalFilters.bilateralfilter(img, f, g, W)
end

function ribnml(img::Array{T},
                W_global::Int,
                W_local::Int,
                h::Real) where {T <: Real}
    img_dst    = Array{T}(undef, size(img)...)
    regions    = LocalFilters.cartesianregion(img)

    W_G_half      = W_global >> 1
    I_G           = CartesianIndex(ntuple(i -> W_G_half, 2))
    window_global = LocalFilters.RectangularBox{2}(-I_G, I_G)

    W_L_half      = W_local >> 1
    I_L           = CartesianIndex(ntuple(i -> W_L_half, 2))
    window_local  = LocalFilters.RectangularBox{2}(-I_L, I_L)

    img_src    = Images.padarray(img, Images.Pad(W_G_half + W_L_half,
                                                 W_G_half + W_L_half))
    k_g_min, k_g_max = LocalFilters.limits(window_global)
    k_l_min, k_l_max = LocalFilters.limits(window_local)

    border_type = "replicate"
    filter_type = ImageFiltering.Algorithm.FIR()
    avg_filter  = Images.OffsetArray(fill(1/(W_local*W_local), W_local, W_local),
                                     -W_L_half:W_L_half,
                                     -W_L_half:W_L_half)
    μ           = ImageFiltering.imfilter(img,  avg_filter, border_type, filter_type)
    prog        = ProgressMeter.Progress(prod(size(img_src)))
    @inbounds for i in regions
        Σwv  = T(0)
        Σw   = T(0)

        x0_idx = LocalFilters.cartesianregion(i - k_l_max, i - k_l_min)
        x0     = img_src[x0_idx]
        μ_x0   = μ[i]

        @simd for j in LocalFilters.cartesianregion(i - k_g_max, i - k_g_min)
            x_idx = LocalFilters.cartesianregion(j - k_l_max, j - k_l_min)
            x     = img_src[x_idx]
            μ_x   = μ[j]

            w    = exp(-0.5*(sum((x - x0).^2) + 2*(μ_x0 - μ_x).^2) / (h*h))
            w    = max(eps(T), w)
            Σw  += w
            Σwv += w*img_src[j]
        end
        img_dst[i] = Σwv / Σw
        ProgressMeter.next!(prog)
    end
    img_dst
end

function hybrid(img::Array,
                W_guided::Int, ϵ::Real,
                W_srbf::Int, σ₁::Real, σ₂::Real,
                W_ribnml_global::Int, W_ribnml_local::Int, h::Real)
#=
    Hybrid algorithm
    
    "A hybrid algorithm for speckle noise reduction of ultrasound images"
    Karamjeet Singh, Sukhjeet Kaur Ranade, Chandan Singh, 
    Computer Methods and Programs in Biomedicine, 2017
=##
    img = deepcopy(img)
    img = guided(img, img, W_guided, ϵ)
    img = srbf(img, W_srbf, σ₁, σ₂)
    img = ribnml(img, W_ribnml_global, W_ribnml_local, h)
end
