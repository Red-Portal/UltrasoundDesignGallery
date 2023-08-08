
using DrWatson
@quickactivate "SpeckleReduction"

include(srcdir("SpeckleReduction.jl"))

import NaturalSort
using BayesianOptimization
using GaussianProcesses
using Distributions
using NLopt

import StatsBase
import ImageQualityIndexes
import PFM
import PNGFiles
import JSON
import PyCall

imio = PyCall.pyimport("imageio")

@eval ImageQualityIndexes begin
    function _ssim_statistics(
        x::GenericImage, ref::GenericImage, window; crop)
        # For RGB and other Color3 images, we don't slide the window at the color channel.
        # In other words, these characters will be calculated channelwisely
        window = kernelfactors(Tuple(repeated(window, ndims(ref))))

        region = map(window, axes(x)) do w, a
            o = length(w) ÷ 2
            # Even if crop=true, it crops only when image is larger than window
            length(a) > length(w) ? (first(a)+o:last(a)-o) : a
        end
        R = crop ? CartesianIndices(region) : CartesianIndices(x)
        alg = ImageFiltering.Algorithm.FIR()

        # don't slide the window in the channel dimension
        μx = view(imfilter(x,   window, "symmetric", alg), R) # equation (14) in [1]
        μy = view(imfilter(ref, window, "symmetric", alg), R) # equation (14) in [1]
        μx² = _mul.(μx, μx)
        μy² = _mul.(μy, μy)
        μxy = _mul.(μx, μy)
        σx² = view(imfilter(_mul.(x,   x  ), window, "symmetric", alg), R) .- μx² # equation (15) in [1]
        σy² = view(imfilter(_mul.(ref, ref), window, "symmetric", alg), R) .- μy² # equation (15) in [1]
        σxy = view(imfilter(_mul.(x,   ref), window, "symmetric", alg), R) .- μxy # equation (16) in [1]

        # after that, channel dimension can be treated generically so we expand them here
        return channelview.((μx², μxy, μy², σx², σxy, σy²))
    end
end

function total_variation(img, mask)
    forwarddiffy(u::AbstractMatrix) = [u[2:end,:]; u[end:end,:]] - u
    forwarddiffx(u::AbstractMatrix) = [u[:,2:end] u[:,end:end]] - u
    ∂img∂x = forwarddiffx(img)
    ∂img∂y = forwarddiffy(img)
    sum(sqrt.(abs.(∂img∂x[mask]).^2 + abs.(∂img∂y[mask]).^2))
end

function q_index(img, masks)
    roi_stats = map(masks) do mask
        μ  = mean(img[mask])
        σ² = var(img[mask])
        μ, σ²
    end

    numerator   = 0
    demonimator = 0
    for i = 1:length(roi_stats)
        for j = 1:length(roi_stats)
            if i == j
                continue
            end
            Δ = roi_stats[i][1] - roi_stats[j][1]
            numerator += Δ*Δ
        end
        demonimator += roi_stats[i][2]
    end
    numerator / demonimator
end

function linear_interpolate(x, x_min, x_max)
    (x_max - x_min)*x + x_min
end

function exp_interpolate(x, x_min, x_max)
    β = log(x_min)
    α = log(x_max / x_min)
    exp(α*x + β)
end

function domain_transform(x)
    #ee1_α_idx  = 1
    ee1_β_idx  = 1
    ee1_σ_idx  = 2
    #ee2_α_idx  = 4
    ee2_β_idx  = 3
    ee2_σ_idx  = 4
    ncd1_s_idx = 5
    ncd1_α_idx = 6
    ncd2_s_idx = 7
    ncd2_α_idx = 8
    rpncd_idx  = 9

    x′ = zeros(9)

    #ee1_α
    x′[ee1_β_idx] = linear_interpolate(x[1], 1.0, 5.0)
    x′[ee1_σ_idx] = exp_interpolate(x[2], 1e-4, 1e-2)

    x′[ee2_β_idx] = exp_interpolate(x[3], 1.0, 5.0)
    x′[ee2_σ_idx] = exp_interpolate(x[4], 1e-4, 1e-2)

    #ncd1_s
    x′[ncd1_s_idx] = linear_interpolate(x[5], 1, 100)
    x′[ncd1_α_idx] = linear_interpolate(x[6], 3e-2, 0.1)
    x′[ncd2_s_idx] = linear_interpolate(x[7], 1, 100)
    x′[ncd2_α_idx] = linear_interpolate(x[8], 3e-2, 0.1)
    x′[rpncd_idx]  = exp_interpolate(x[9],  1e-4, 1e-2)
    x′
end

# function main1()
#     img_noisy = PFM.pfmread("../../../data/phantom/phantom_ATS549_2_1.pfm")
#     mask      = PNGFiles.load("../../../data/phantom/phantom_ATS549_2_1.png")
#     masks     = Array{Matrix{Bool}}(undef, 6)
#     masks[1]  = PNGFiles.load("../../../data/phantom/mask1.png") .> 0
#     masks[2]  = PNGFiles.load("../../../data/phantom/mask2.png") .> 0
#     masks[3]  = PNGFiles.load("../../../data/phantom/mask3.png") .> 0
#     masks[4]  = PNGFiles.load("../../../data/phantom/mask4.png") .> 0
#     masks[5]  = PNGFiles.load("../../../data/phantom/mask5.png") .> 0
#     masks[6]  = PNGFiles.load("../../../data/phantom/mask6.png") .> 0

#     q0 = q_index(img_noisy, masks)

#     img_row_maj     = Array(img_noisy')
#     mask_row_maj    = Array(mask')
#     img_out_row_maj = deepcopy(img_row_maj)
#     mask_bool       = convert(Array{Bool}, Float32.(mask) .> 0)

#     f(x) = begin
#         x′      = domain_transform(x)
#         x′      = Float32.(x′)
#         ccall((:process_image_c_api, "../../../build/libipcore"),
#               Cvoid,
#               (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}),
#               img_row_maj, mask_row_maj,
#               size(img_noisy, 1), size(img_noisy, 2),
#               x′, img_out_row_maj)
#         img_out = Array(img_out_row_maj')
#         q_hat = q_index(img_out, masks) / q0
#         ssim  = ImageQualityIndexes.assess_ssim(img_out, img_noisy)
#         q_hat + 15*ssim
#     end

#     usdg_log = JSON.parsefile("../../../build/q_index_optim_15.json")
#     for log_entry in usdg_log[1]
#         println(f(log_entry["x"]), ",")
#     end
# end

struct NormalIterator
    N::Int
end

Base.length(it::NormalIterator) = it.N

function Base.iterate(it::NormalIterator, s=1)
    s == it.N + 1 && return nothing
    rand(9), s+1
end

function metric_ssnr(img, mask)
    μ = mean(img[mask])
    σ = std(img[mask])
    μ / σ
end

function gcnr(roi1, roi2)
    x_bins    = collect(0:0.001:1)
    roi1_hist = StatsBase.fit(StatsBase.Histogram, roi1, x_bins)
    roi2_hist = StatsBase.fit(StatsBase.Histogram, roi2, x_bins)

    roi1_hist = StatsBase.normalize(roi1_hist)
    roi2_hist = StatsBase.normalize(roi2_hist)
    avl       = mean([min(roi1_hist.weights[idx], roi2_hist.weights[idx])
                      for idx in 1:length(x_bins)-1])
    1 - avl
end

function edge_resol(img, center_x, center_y, r, μ1, μ2)
    Σedge = 0
    degs  = vcat(-10:10, 180 .+ -10:10)
    for deg = degs
        θ    = deg/180*π
        x    = round.(Int, center_x .+ cos(θ)*(0:1:r))
        y    = round.(Int, center_y .+ sin(θ)*(0:1:r))

        line = [img[x[i], y[i]] for i = 1:length(x)]
        q2   = abs(μ1 - μ2)*0.2 + μ1
        q8   = abs(μ1 - μ2)*0.8 + μ1
        idx2 = findlast(idx -> line[idx] < q2, 1:length(line))
        idx8 = findfirst(idx -> line[idx] > q8, 1:length(line))
        edge = abs(idx2 - idx8)

        Σedge += edge
    end
    Σedge / length(degs)
end

function metric_cnr(roi1, roi2)
    μ1  = mean(roi1)
    σ²1 = var(roi1)
    μ2  = mean(roi2)
    σ²2 = var(roi2)
    cnr = abs(μ1 - μ2) / sqrt(σ²1 + σ²2)
    #cnr = 10*log10(cnr)
end

function main()
    img_noisy = PFM.pfmread("../../../data/phantom/cyst_phantom.pfm")
    mask      = trues(size(img_noisy))

    roi_mask1 = PNGFiles.load("../../../data/phantom/cyst_mask_in.png")  .> 0
    roi_mask2 = PNGFiles.load("../../../data/phantom/cyst_mask_out.png") .> 0

    #img_noisy = PFM.pfmread("../../../data/phantom/phantom_ATS549_2_1.pfm")
    #img_noisy = PFM.pfmread("../../../data/selections/liver1/subject2_liver2_1.pfm")
    #mask      = PNGFiles.load("../../../data/selections/liver1/subject2_liver2_1.png")
    mask      = mask .> 0

    #roi_mask = falses(size(mask))
    #roi_mask[250:340, 250:370] .= true

    #mask      = PNGFiles.load("../../../data/phantom/phantom_ATS549_2_1.png")
    # masks     = Array{Matrix{Bool}}(undef, 6)
    # masks[1]  = PNGFiles.load("../../../data/phantom/mask1.png") .> 0
    # masks[2]  = PNGFiles.load("../../../data/phantom/mask2.png") .> 0
    # masks[3]  = PNGFiles.load("../../../data/phantom/mask3.png") .> 0
    # masks[4]  = PNGFiles.load("../../../data/phantom/mask4.png") .> 0
    # masks[5]  = PNGFiles.load("../../../data/phantom/mask5.png") .> 0
    # masks[6]  = PNGFiles.load("../../../data/phantom/mask6.png") .> 0

    #q0 = q_index(img_noisy, masks)

    img_row_maj     = Array(img_noisy')
    mask_row_maj    = Array(mask')
    img_out_row_maj = deepcopy(img_row_maj)
    mask_bool       = convert(Array{Bool}, Float32.(mask) .> 0)
    μ1_noisy        = mean(img_noisy[roi_mask1])
    μ2_noisy        = mean(img_noisy[roi_mask2])

    f(x) = begin
        x′      = domain_transform(x)
        x′      = Float32.(x′)
        ccall((:process_image_c_api, "../../../build/libipcore"),
              Cvoid,
              (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}),
              img_row_maj, mask_row_maj,
              size(img_noisy, 1), size(img_noisy, 2),
              x′, img_out_row_maj)
        img_out = Array(img_out_row_maj')

        cnr  = metric_cnr(img_out[roi_mask1], img_out[roi_mask2])
        edge = edge_resol(img_out, 146, 187, 80, μ1_noisy, μ2_noisy)

        @info "" cnr=cnr edge=edge
        #cnr - 10*edge
        -edge
    end

    display(Plots.plot(img_noisy[180,:]))
    #display(Plots.histogram!(reshape(img_noisy[roi_mask2], :)))
    #return

    # f_wrapper(x, grad) = begin
    #     y = f(x)
    #     println(y)
    #     y
    # end

    # opt = Opt(:LN_NELDERMEAD, 9)
    # opt.lower_bounds = zeros(9)
    # opt.upper_bounds = ones(9)
    # opt.maxeval = 24
    # opt.max_objective = f_wrapper
    # println(optimize(opt, rand(9)))


    # Choose as a model an elastic GP with input dimensions 2.
    # The GP is called elastic, because data can be appended efficiently.
    n_dims = 9
    model  = ElasticGPE(n_dims,                            # 2 input dimensions
                        mean = MeanZero(),         
                        kernel = SE(1.0, 1.),
                        logNoise = -4,
                        capacity = 3000)              # the initial capacity of the GP is 3000 samples.

    # Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
    modeloptimizer = MAPGPOptimizer(every = 10, noisebounds = [-4, -3],       # bounds of the logNoise
                                    kernbounds = [[-4, -4], [4, 8]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                    maxeval = 1000)

    #modeloptimizer = NoModelOptimizer()
    opt = BOpt(f,
               model,
               ExpectedImprovement(), # type of acquisition
               modeloptimizer,                        
               zeros(n_dims), ones(n_dims),              # lowerbounds, upperbounds         
               repetitions = 1,                          # evaluate the function for each input 5 times
               maxiterations = 500,                   # evaluate at 100 input positions
               sense = Max,                              # minimize the function
               acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                     restarts = 128,       # run the NLopt method from 5 random initial conditions each time.
                                     maxeval = 10000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
               initializer = ScaledSobolIterator(zeros(n_dims), ones(n_dims), 32),
               verbosity = Progress)

    result = boptimize!(opt)
    display(result)
    x′     = domain_transform(result.observed_optimizer)
    x′     = Float32.(x′)
    println(f(result.observed_optimizer))
    println(f(result.model_optimizer))

    x′2 = domain_transform(result.model_optimizer)
    x′2 = Float32.(x′2)

    ccall((:process_image_c_api, "../../../build/libipcore"),
          Cvoid,
          (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}),
          img_row_maj, mask_row_maj,
          size(img_noisy, 1), size(img_noisy, 2),
          x′, img_out_row_maj)
    img_out = Array(img_out_row_maj')

    ccall((:process_image_c_api, "../../../build/libipcore"),
          Cvoid,
          (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}),
          img_row_maj, mask_row_maj,
          size(img_noisy, 1), size(img_noisy, 2),
          x′2, img_out_row_maj)
    img_out2 = Array(img_out_row_maj')

    #q_hat = q_index(img_out, masks) / q0
    #ssim  = ImageQualityIndexes.assess_ssim(img_noisy, img_out)
    #metric_ssnr()
    #@info "result" q̂=q_hat ssim=ssim obj=q_hat + 15*ssim

    display(Plots.plot(  img_noisy[181, 330:end]))
    display(Plots.plot!( img_out[181, 330:end]))
    #display(Plots.plot( img_noisy[146, 187:280]))
    #display(Plots.plot!(img_out[  146, 187:280]))
    #display(Plots.hline!([ mean(img_noisy[roi_mask1]), mean(img_noisy[roi_mask2]) ]))

    #return img_out
    img_out   = adjust_dynamic_range(img_out,   0.01, 0.8)
    img_out2  = adjust_dynamic_range(img_out2,  0.01, 0.8)
    img_noisy = adjust_dynamic_range(img_noisy, 0.01, 0.8)
    ImageView.imshow(img_noisy, name="base")
    ImageView.imshow(img_out,   name="best obs")
    ImageView.imshow(img_out2,  name="best model")
end

function process_image(img, mask, param)
    img_row_maj     = Array(img')
    mask_row_maj    = Array(mask')
    img_out_row_maj = deepcopy(img_row_maj)

    param = Float32.(param)
    ccall((:process_image_c_api, "../../../build/libipcore"),
          Cvoid,
          (Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}),
          img_row_maj, mask_row_maj,
          size(img, 1), size(img, 2),
          param, img_out_row_maj)
    img_out = Array(img_out_row_maj')
end

function load_images_sorted(path)
    fnames   = filter(fname -> occursin(".pfm", fname), readdir(path, join=true))
    fnames   = NaturalSort.sort(fnames, lt=NaturalSort.natural)
    sequence = imio.imread.(fnames)
end

function load_mask(path)
    mask = filter(fname -> occursin(".png", fname), readdir(path, join=true))[1]
    mask = PNGFiles.load(mask)
    mask = mask .> 0
end

function measure()
    img_noisy = PFM.pfmread("../../../data/phantom/cyst_phantom.pfm")
    mask      = trues(size(img_noisy))
    roi_mask1 = PNGFiles.load("../../../data/phantom/cyst_mask_in.png")  .> 0
    roi_mask2 = PNGFiles.load("../../../data/phantom/cyst_mask_out.png") .> 0

    function show(file, name)
        img_out = PFM.pfmread(file)
        img_out = img_out[1:size(roi_mask1,1), 1:size(roi_mask1,2)]
        cnr     = metric_cnr(img_out[roi_mask1], img_out[roi_mask2])
        edge    = edge_resol(img_out, 146, 187, 80, mean(img_out[roi_mask1]), mean(img_out[roi_mask2]))
        @info name cnr=Float64(cnr) edge=edge
    end
    show("../../../data/results/phantom/phantom_osrad.pfm",  "osrad")
    show("../../../data/results/phantom/phantom_nllr.pfm",   "nllr")
    show("../../../data/results/phantom/phantom_pfdtv.pfm",  "pfdtv")
    show("../../../data/results/phantom/phantom_mnlm.pfm",   "mnlm")
    show("../../../data/results/phantom/phantom_lpndsf.pfm", "lpndsf")
    show("../../../data/results/phantom/phantom_admss.pfm",  "admss")
end

function main2()
    param = [
        2.0989457873516386,
        0.0014419932799235446,
        1.1550296676793073,
        0.00028796634737797856,
        78.13109730077265,
        0.08139456670580981,
        71.2234147811635,
        0.09974046971443012,
        0.00010000000000000009,
    ]
    imgs = load_images_sorted("../../../data/selections/liver2/")   
    mask = load_mask(         "../../../data/selections/liver2/")  
    ProgressMeter.@showprogress map(enumerate(imgs)) do (idx, img)
        img     = img[end:-1:1,:] 
        img_out = process_image(img, mask, param)
        PFM.pfmwrite("../../../data/results/liver2_clpdQ/processed_$(idx).pfm", img_out)
    end
end
