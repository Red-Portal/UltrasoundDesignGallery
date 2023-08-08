
using DrWatson
@quickactivate "SpeckleReduction"

import NaturalSort
import PNGFiles
import PyCall
import ImageQualityIndexes
import Printf
import PrettyTables
import StatsBase
import PNGFiles

include(srcdir("SpeckleReduction.jl"))

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
        μx = view(imfilter(x,   window, "symmetric"), R) # equation (14) in [1]
        μy = view(imfilter(ref, window, "symmetric"), R) # equation (14) in [1]
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

function run_dpad(logcomp, mask; kwargs...)
    mask    = Images.Gray.(mask) 
    mask    = Float32.(mask) .> 0

    Δt      = 1.0 
    n_iters = 30
    window  = 5

    envelope = exp10.(logcomp)
    img_out  = dpad(envelope, Δt, n_iters, window; mask=mask, robust=false)
    img_out[img_out .< 1e-15] .= 1e-15
    img_out = log10.(img_out)
    img_out = clamp.(img_out, 0, 1)
    img_out[.!mask] .= 0.0
    img_out
end

function run_osrad(logcomp, mask; kwargs...)
    mask    = Images.Gray.(mask) 
    mask    = Float32.(mask) .> 0

    Δt      = 1.0 
    n_iters = 30
    window  = 5
    ctang   = 0.1

    envelope = exp10.(logcomp)
    img_out  = osrad(envelope, Δt, n_iters, ctang, window; mask=mask, robust=false)
    img_out[img_out .< 1e-15] .= 1e-15
    img_out = log10.(img_out)
    img_out = clamp.(img_out, 0, 1)
    img_out[.!mask] .= 0.0
    img_out
end

# function run_hybrid(logcomp, mask)
#     W_guided        = 5
#     ϵ               = 650*36
#     W_srbf          = 5
#     σ₁              = 13.65
#     σ₂              = 75
#     W_ribnml_global = 11
#     W_ribnml_local  = 5
#     h               = 14.3
#     img_out = hybrid(logcomp, W_guided, ϵ, W_srbf, σ₁, σ₂,
#                      W_ribnml_global, W_ribnml_local, h)
#     img_out = clamp.(img_out, 0, 1)
#     img_out[.!mask] .= 0.0
#     img_out
# end

function run_pfdtv(logcomp, mask; kwargs...)
    logcomp *= 255

    n_iters  = 20
    s_min    = 1
    s_max    = 15

    ratio      = 0.4
    itpl       = BSpline(Cubic(Line(OnGrid())))
    img_scaled = Images.imresize(logcomp; ratio=ratio, method=itpl)
    mask       = Images.imresize(mask; ratio=ratio) .> 0

    img_out = pfdtv(img_scaled, n_iters, s_min, s_max,  mask=mask)
    img_out = Images.imresize(img_out; ratio=1/ratio, method=itpl)
    img_out = clamp.(img_out, 0, 255) / 255
end

function run_mnlm(logcomp, mask; kwargs...)
    search_size = 11
    patch_size  = 3
    n_scales    = 3
    h           = kwargs[:mnlm_h]

    img_out = mnml(logcomp, search_size, patch_size, n_scales, h; mask=mask)
    img_out = clamp.(img_out, 0, 1)
    img_out
end

function run_lpndsf(logcomp, mask; kwargs...)
    Δt        = 0.2
    mse_thres = 0.00005
    r         = [0.1, 5.0, 0.1]
    k         = [0.3, 0.1, 0.1]

    DR       = log10(2^14)
    logcomp *= DR
    img_out  = lpndsf(logcomp, Δt, mse_thres, k, r; mask=mask)
    img_out /= DR
    img_out  = clamp.(img_out, 0, 1)
    img_out[.!mask] .= 0.0
    img_out
end

function run_cpld(logcomp, mask; kwargs...)
    ee1_α = 2.0
    ee1_β = 4.0
    ee1_σ = 0.01

    ee2_α = 2.0
    ee2_β = 1.3
    ee2_σ = 0.0003

    ncd1_s = 3
    ncd1_α = 0.06

    ncd2_s = 40
    ncd2_α = 0.05

    rpncd_k = 0.001

    img_out = clpd(logcomp,
                   ee1_α, ee1_β, ee1_σ,
                   ee2_α, ee2_β, ee2_σ,
                   ncd1_s, ncd1_α,
                   ncd2_s, ncd2_α,
                   rpncd_k;
                   mask=mask)
    img_out  = clamp.(img_out, 0, 1)
end

function apply_algorithm(inpath, algorithm, outpath; kwargs...)
    n_avg    = 2
    sequence = filter(fname -> occursin(".pfm", fname), readdir(inpath, join=true))
    mask     = filter(fname -> occursin(".png", fname), readdir(inpath, join=true))[1]
    mask     = PNGFiles.load(mask)
    mask     = mask .> 0

    if (!isdir(outpath))
        mkdir(outpath)
    end
    sequence = NaturalSort.sort(sequence, lt=NaturalSort.natural)
    frames   = map(fname -> PFM.pfmread(fname), sequence)
    n_frames = length(frames)
    frames   = map(1:n_frames) do idx
        mean(frames[idx:min(idx + n_avg - 1, n_frames)])
    end

    map(1:n_frames) do idx
        outname = joinpath(outpath, "processed_$(idx).pfm")
        if isfile(outname)
            @info "skipping $(outname)"
        else
            img_out = algorithm(frames[idx], mask; kwargs...)
            PFM.pfmwrite(outname, img_out)
        end
    end
end

function run_algorithms(inpath, outpath; kwargs...) 
    apply_algorithm(inpath, run_osrad,  outpath * "_osrad";  kwargs...)
    apply_algorithm(inpath, run_pfdtv,  outpath * "_pfdtv";  kwargs...)
    apply_algorithm(inpath, run_mnlm,   outpath * "_mnlm";   kwargs...)
    apply_algorithm(inpath, run_lpndsf, outpath * "_lpndsf"; kwargs...)
end

function export_result(inpath, frame, reject_level, dynamic_range, outpath)
    fnames   = filter(fname -> occursin(".pfm", fname), readdir(inpath, join=true))
    fnames   = NaturalSort.sort(fnames, lt=NaturalSort.natural)
    sequence = imio.imread.(fnames)
    img      = adjust_dynamic_range(sequence[frame], reject_level, dynamic_range)
    img      = img[end:-1:1,:] 
    PNGFiles.save(outpath * ".png", img)
end

function export_results(basepath, inpath, frame, reject_level, dynamic_range, outpath)
    export_result(basepath, frame, reject_level, dynamic_range, outpath)
    #export_result(inpath * "_clpdQ",  frame, reject_level, dynamic_range, outpath * "_clpdQ")
    export_result(inpath * "_clpda",  frame, reject_level, dynamic_range, outpath * "_clpda")
    export_result(inpath * "_clpdb",  frame, reject_level, dynamic_range, outpath * "_clpdb")
    #try 
        #%export_result(inpath * "_clpdc",  frame, reject_level, dynamic_range, outpath * "_clpdc")
        #export_result(inpath * "_clpdd",  frame, reject_level, dynamic_range, outpath * "_clpdd")
    #catch
        export_result(inpath * "_clpde",  frame, reject_level, dynamic_range, outpath * "_clpde")
        export_result(inpath * "_clpdf",  frame, reject_level, dynamic_range, outpath * "_clpdf")
    #end

    try
        export_result(inpath * "_osrad",  frame, reject_level, dynamic_range, outpath * "_osrad")
        export_result(inpath * "_admss",  frame, reject_level, dynamic_range, outpath * "_admss")
        export_result(inpath * "_lpndsf", frame, reject_level, dynamic_range, outpath * "_lpndsf")
        export_result(inpath * "_pfdtv",  frame, reject_level, dynamic_range, outpath * "_pfdtv")
        export_result(inpath * "_nllr",   frame, reject_level, dynamic_range, outpath * "_nllr")
        export_result(inpath * "_mnlm",   frame, reject_level, dynamic_range, outpath * "_mnlm")
    catch
        try
        export_result(inpath * "_osrad",  2, reject_level, dynamic_range, outpath * "_osrad")
        export_result(inpath * "_admss",  2, reject_level, dynamic_range, outpath * "_admss")
        export_result(inpath * "_lpndsf", 2, reject_level, dynamic_range, outpath * "_lpndsf")
        export_result(inpath * "_pfdtv",  2, reject_level, dynamic_range, outpath * "_pfdtv")
        export_result(inpath * "_nllr",   2, reject_level, dynamic_range, outpath * "_nllr")
        export_result(inpath * "_mnlm",   2, reject_level, dynamic_range, outpath * "_mnlm")
        catch
        export_result(inpath * "_osrad",  1, reject_level, dynamic_range, outpath * "_osrad")
        export_result(inpath * "_admss",  1, reject_level, dynamic_range, outpath * "_admss")
        export_result(inpath * "_lpndsf", 1, reject_level, dynamic_range, outpath * "_lpndsf")
        export_result(inpath * "_pfdtv",  1, reject_level, dynamic_range, outpath * "_pfdtv")
        export_result(inpath * "_nllr",   1, reject_level, dynamic_range, outpath * "_nllr")
        export_result(inpath * "_mnlm",   1, reject_level, dynamic_range, outpath * "_mnlm")
        end
    end
end

function metric_ssnr(img, mask)
    μ = mean(img[mask])
    σ = std(img[mask])
    20*log10(μ) - 20*log10(σ)
end

function metric_beta(img_base, img, mask)
#=
    X. Hao, S. Gao, and X. Gao, 
    “A novel multiscale nonlinear thresholding method for ultrasound speckle suppressing,” 
    IEEE Trans. Med. Imag., vol. 18, pp. 787–794, Sep. 1999.
=##
    ∇²a = ImageFiltering.imfilter(img_base, ImageFiltering.Laplacian())
    ∇²b = ImageFiltering.imfilter(img,      ImageFiltering.Laplacian())
    ∇²a = reshape(∇²a[mask], :)
    ∇²b = reshape(∇²b[mask], :)
    dot(∇²a, ∇²b) / (norm(∇²b) * norm(∇²a))
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

function metric_cnr(roi1, roi2)
    μ1  = mean(roi1)
    σ²1 = var(roi1)
    μ2  = mean(roi2)
    σ²2 = var(roi2)
    cnr = abs(μ1 - μ2) / sqrt(σ²1 + σ²2)
    cnr = 10*log10(cnr)
end

function metric_gcnr(roi1, roi2)
    x_bins    = collect(0:0.001:1)
    roi1_hist = StatsBase.fit(StatsBase.Histogram, roi1, x_bins)
    roi2_hist = StatsBase.fit(StatsBase.Histogram, roi2, x_bins)

    roi1_hist = StatsBase.normalize(roi1_hist)
    roi2_hist = StatsBase.normalize(roi2_hist)
    avl       = mean([min(roi1_hist.weights[idx], roi2_hist.weights[idx])
                      for idx in 1:length(x_bins)-1])
    1 - avl
end

function evaluate_liver1_metrics(name, base_path, frame_idx, filtered_path)
    base_imgs     = load_images_sorted(base_path)[frame_idx]
    filtered_imgs = load_images_sorted(filtered_path)[frame_idx]
    mask          = load_mask(base_path)

    roi_mask = falses(size(mask))
    roi_mask[250:340, 250:370] .= true

    ssnrs = metric_ssnr.(filtered_imgs, Ref(roi_mask))
    ssims = map(1:length(base_imgs)) do i
        ImageQualityIndexes.assess_ssim(filtered_imgs[i], base_imgs[i])
    end

    ssnr_10, ssnr_90 = quantile(ssnrs, [0.1, 0.9])
    ssim_10, ssim_90 = quantile(ssims, [0.1, 0.9])

    ssnr_str = Printf.@sprintf("%2.1f (%2.1f, %2.1f)", mean(ssnrs), ssnr_10, ssnr_90)
    ssim_str = Printf.@sprintf("%.3f (%.3f, %.3f)", mean(ssims), ssim_10, ssim_90)
    @info "$(name)\n SSRN = $(ssnr_str) \n SSIM = $(ssim_str)"
    [name, ssnr_str, ssim_str]
end

function evaluate_liver1_metrics_all()
    base_img_path = "../../../data/selections/liver1"

    table        = Array{String}(undef, 11, 3)
    table[1,  :] = evaluate_liver1_metrics("OSRAD",  base_img_path, 1:16, "../../../data/results/liver1_osrad")
    table[2,  :] = evaluate_liver1_metrics("ADMSS",  base_img_path, 1:16, "../../../data/results/liver1_admss")
    table[3,  :] = evaluate_liver1_metrics("LPNDSF", base_img_path, 1:16, "../../../data/results/liver1_lpndsf")
    table[4,  :] = evaluate_liver1_metrics("MNLM",   base_img_path, 1:16, "../../../data/results/liver1_mnlm")
    table[5,  :] = evaluate_liver1_metrics("NLLR",   base_img_path, 1:16, "../../../data/results/liver1_nllr")
    table[6,  :] = evaluate_liver1_metrics("PFDTV",  base_img_path, 1:16, "../../../data/results/liver1_pfdtv")
    #table[7,  :] = evaluate_liver1_metrics("CLPD-obj.", base_img_path, 1:16, "../../../data/results/liver1_clpdQ")
    table[8,  :] = evaluate_liver1_metrics("CLPD-A", base_img_path, 1:16, "../../../data/results/liver1_clpda")
    table[9,  :] = evaluate_liver1_metrics("CLPD-B", base_img_path, 1:16, "../../../data/results/liver1_clpdb")
    table[10, :] = evaluate_liver1_metrics("CLPD-C", base_img_path, 1:16, "../../../data/results/liver1_clpdc")
    table[11, :] = evaluate_liver1_metrics("CLPD-D", base_img_path, 1:16, "../../../data/results/liver1_clpdd")
    # display(PrettyTables.pretty_table(table;
    #                                   backend = Val(:latex),
    #                                   tf = PrettyTables.tf_latex_booktabs,
    #                                   header=(["Algorithm", "SSNR", "SSIM"], ["[dB]", ""])))
    table
end

function evaluate_cardiac3_metrics(name, base_path, frame_idx, filtered_path; flip=false)
    base_imgs = load_images_sorted(base_path)[frame_idx]
    base_imgs = [base_img[size(base_img,1):-1:1, 1:size(base_img,2)] for base_img in base_imgs]
    base_imgs = [Float32.(base_img) for base_img in base_imgs]

    #files = readdir(filtered_path, join=true)
    imgs  = load_images_sorted(filtered_path)[frame_idx]
    imgs  = [img[1:size(base_imgs[1],1), 1:size(base_imgs[1],2)] for img in imgs]
    imgs  = if flip
        [img[size(img,1):-1:1, 1:size(base_imgs[1],2)] for img in imgs]
    else
        imgs
    end
    #roi_masks = FileIO.load("patient0133_4CH_ES_gt.mhd")[:,:,1]
    myo_lv    = FileIO.load("myo_lv_mask.png") .> 0
    blood_lv  = FileIO.load("blood_lv_mask.png") .> 0
    #myo_la    = (0.010  .< roi_masks) .& (roi_masks .< 0.013)
    #blood_lv  = (0.0038 .< roi_masks) .& (roi_masks .< 0.005)

    #myo_lv   = Images.imresize(myo_lv',   size(roi_masks)) .> 0
    #myo_la   = Images.imresize(myo_la',   size(roi_masks)) .> 0
    #blood_lv = Images.imresize(blood_lv', size(roi_masks)) .> 0

    #ImageView.imshow(myo_lv .* imgs[1]; name="myo")
    #ImageView.imshow(blood_lv .* imgs[1]; name="blood")
    #ImageView.imshow(imgs[1])
    #ImageView.imshow(base_img[1])
    #display(Plots.histogram(imgs[1][myo_lv]))
    #display(Plots.histogram!(imgs[1][blood_lv]))
    #throw()
    #quit()

    gcnr  = metric_gcnr(imgs[1][myo_lv], imgs[1][blood_lv])
    cnr   = metric_cnr( imgs[1][myo_lv], imgs[1][blood_lv])
    ssnr  = metric_ssnr(imgs[1], myo_lv)
    ssims = map(imgs, base_imgs) do img, base_img
        ImageQualityIndexes.assess_ssim(img, base_img)
    end
    ssim_10, ssim_90 = quantile(ssims, [0.1, 0.9])

    gcnr_str = Printf.@sprintf("%.3f",  gcnr)
    cnr_str  = Printf.@sprintf("%1.2f", cnr)
    ssnr_str = Printf.@sprintf("%1.2f", ssnr)
    ssim_str = Printf.@sprintf("%.3f {\\tiny{(%.3f, %.3f)}}",  mean(ssims), ssim_10, ssim_90)
    @info "$(name)\n gCNR = $(gcnr_str) \n CNR = $(cnr_str) \n SSNR = $(ssnr_str) \n SSIM = $(ssim_str)"
    [name, gcnr_str, cnr_str, ssnr_str, ssim_str]
end

function evaluate_cardiac3_metrics_all()
    base_img_path = "../../../data/selections/cardiac3"

    table        = Array{String}(undef, 10, 5)
    table[1,  :] = evaluate_cardiac3_metrics("OSRAD",  base_img_path, 1:21, "../../../data/results/cardiac3_osrad",  flip=true)
    table[2,  :] = evaluate_cardiac3_metrics("ADMSS",  base_img_path, 1:21, "../../../data/results/cardiac3_admss",  flip=true)
    table[3,  :] = evaluate_cardiac3_metrics("LPNDSF", base_img_path, 1:21, "../../../data/results/cardiac3_lpndsf", flip=true)
    table[4,  :] = evaluate_cardiac3_metrics("MNLM",   base_img_path, 1:21, "../../../data/results/cardiac3_mnlm",   flip=true)
    table[5,  :] = evaluate_cardiac3_metrics("NLLR",   base_img_path, 1:21, "../../../data/results/cardiac3_nllr",   flip=true)
    table[6,  :] = evaluate_cardiac3_metrics("PFDTV",  base_img_path, 1:21, "../../../data/results/cardiac3_pfdtv",  flip=true)
    table[7,  :] = evaluate_cardiac3_metrics("CLPD-A", base_img_path, 1:21, "../../../data/results/cardiac3_clpda",  flip=true)
    table[8,  :] = evaluate_cardiac3_metrics("CLPD-B", base_img_path, 1:21, "../../../data/results/cardiac3_clpdb",  flip=true)
    table[9,  :] = evaluate_cardiac3_metrics("CLPD-E", base_img_path, 1:21, "../../../data/results/cardiac3_clpde",  flip=true)
    table[10, :] = evaluate_cardiac3_metrics("CLPD-F", base_img_path, 1:21, "../../../data/results/cardiac3_clpdf",  flip=true)
    # display(PrettyTables.pretty_table(table;
    #                                   backend = Val(:latex),
    #                                   tf = PrettyTables.tf_latex_booktabs,
    #                                   header=(["Algorithm", "SSNR", "SSIM"], ["[dB]", ""])))
    table
end

function main(mode)
    if mode == :run_algorithms
        for name ∈ ["liver1", "liver2"]
            inpath  = "../../../data/selections/$(name)"
            outpath = "../../../data/results/$(name)"
            run_algorithms(inpath, outpath; mnlm_h=0.12)
        end
        for name ∈ ["cardiac1", "cardiac2", "cardiac3"]
            inpath  = "../../../data/selections/$(name)"
            outpath = "../../../data/results/$(name)"
            run_algorithms(inpath, outpath; mnlm_h=0.2)
        end
    elseif (mode == :export)
        # export_results("../../../data/selections/liver1/",
        #                "../../../data/results/liver1",
        #                6, 0.18, 0.5,
        #                "../../../paper/tpami/figures/liver1")
        # export_results("../../../data/selections/liver2/",
        #                "../../../data/results/liver2",
        #                1, 0.15, 0.5,
        #                "../../../paper/tpami/figures/liver2")
        #export_results("../../../data/results/liver2",    1, 0.15, 0.5, "../../../paper/tpami/figures/liver2")
        # export_results("../../../data/selections/cardiac1/",
        #                "../../../data/results/cardiac1",
        #                30, 0.10, 0.8,
        #                "../../../paper/tpami/figures/cardiac1")
        export_results("../../../data/selections/cardiac3/",
                       "../../../data/results/cardiac3",
                       1, 0.05, 0.8,
                       "../../../paper/tpami/figures/cardiac3")
    elseif mode == :liver1_metrics
        evaluate_liver1_metrics_all()
    elseif mode == :cardiac3_metrics
        evaluate_cardiac3_metrics_all()
    end

    #export_results("../../../data/results/liver1", 6, 0.15, 0.5, "../../../paper/tpami/figures/liver1")

    # logcomp = PFM.pfmread("../../../data/selections/cardiac1/cardiac1_2.pfm")
    # mask    = PNGFiles.load("../../../data/selections/cardiac1/cardiac1_1.png")
    # rej     = 0.15
    # DR      = 0.5

    # logcomp = PFM.pfmread("../../../data/selections/liver1/subject2_liver2_6.pfm")
    # mask    = PNGFiles.load("../../../data/selections/liver1/subject2_liver2_1.png")
    # rej     = 0.15
    # DR      = 0.5

    # run_algorithms(logcomp, mask, rej, DR, "../../../paper/tpami/figures/liver1")

    # logcomp = PFM.pfmread("../../../data/subjects/log/subject2_liver1_1.pfm")
    # mask    = PNGFiles.load("../../../data/subjects/masks/subject2_liver1_1.png")
    # rej     = 0.15
    # DR      = 0.5
    #mask    = mask .> 0

    #run_algorithms(logcomp, mask, rej, DR, "../../../paper/tpami/figures/liver2")

    #img_out = run_osrad(logcomp, mask)
    #img_out = run_pfdtv(logcomp, mask)
    #img_out = run_hybrid(logcomp, mask)
    #img_out = run_mnml(logcomp, mask)
    #img_out = run_lpndsf(logcomp, mask)
    # img_out = run_cpld(logcomp, mask)

    # img_base = adjust_dynamic_range(logcomp, rej, DR)
    # img_out  = adjust_dynamic_range(img_out, rej, DR)
    # view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
    #                               ImageCore.colorview(Images.Gray, img_out);
    #                               nrow=1,
    #                               npad=10)
    # ImageView.imshow(view)
end

