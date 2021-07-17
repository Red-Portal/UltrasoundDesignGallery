
function pnad(img, Δt, n_iters, σ, σ_g::Array, ctang::Array;
              gmm=nothing,
              n_classes=0,
              mask=trues(size(img)...))
    M = size(img, 1)
    N = size(img, 2)
    C = n_classes

    img_src = deepcopy(img)
    img_dst = Array{Float32}(undef, M, N)

    img_σ      = Array{Float32}(undef, M, N)
    D_xx_bayes = zeros(M, N)
    D_xy_bayes = zeros(M, N)
    D_yy_bayes = zeros(M, N)

    G_x      = Array{Float32}(undef, M, N)
    G_y      = Array{Float32}(undef, M, N)
    G_post_x = Array{Float32}(undef, M, N, C)
    G_post_y = Array{Float32}(undef, M, N, C)
    #G_post_x_σ = Array{Float32}(undef, M, N, C)
    #G_post_y_σ = Array{Float32}(undef, M, N, C)

    k = floor(Int, max(σ, 1)*6/2)*2 + 1

    if (isnothing(gmm))
        pixels = reshape(Float64.(img_src[mask]), :)
        pixels = reshape(pixels, (:,1))

        gmm = GaussianMixtures.GMM(n_classes, pixels, nIter=0)

        GaussianMixtures.em!(gmm, pixels; nIter=16, varfloor=1e-7)
        idx    = sortperm(gmm.μ[:,1]; rev=true)
        gmm.μ  = gmm.μ[idx,:]
        gmm.Σ  = gmm.Σ[idx,:]
        gmm.w  = gmm.w[idx]
    end

    ImageFiltering.imfilter!(img_σ, img_src,
                             ImageFiltering.Kernel.gaussian((σ,σ), (k,k)),
                             "replicate",
                             ImageFiltering.Algorithm.FIR())

    posterior, _ = GaussianMixtures.gmmposterior(gmm, reshape(img_σ, (:,1)))
    posterior    = reshape(posterior, (size(img_src)..., size(posterior, 2)))

    @inbounds for c = 1:C
        @inbounds for j = 1:N
            @inbounds for i = 1:M
                xm = max(i-1, 1)
                xp = min(i+1, M)
                ym = max(j-1, 1)
                yp = min(j+1, N)

                G_post_x[i,j,c] = (posterior[xp, j,  c] - posterior[xm, j,  c])/2
                G_post_y[i,j,c] = (posterior[i,  yp, c] - posterior[i,  ym, c])/2
                G_x[i,j]        = (img_σ[xp, j ] - img_σ[xm, j ])/2
                G_y[i,j]        = (img_σ[i,  yp] - img_σ[i,  ym])/2
            end
        end
    end

    fuck = zeros(M,N,C)

    for c = 1:C
        for j = 1:N
            for i = 1:M
                v1x = G_post_x[i,j,c]
                v1y = G_post_y[i,j,c]

                #G_mag = sqrt(v1x*v1x + v1y*v1y)
                #G_mag = G_x[i,j]*G_x[i,j] +  #G_x[i,j]*v1x + G_y[i,j]*v1y
                G_mag = abs(G_x[i,j]*v1x + G_y[i,j]*v1y)
                λ1    = lorentzian_errornorm(abs(G_mag), σ_g[c])
                λ2    = ctang[c]

                fuck[i,j,c] = λ1

                v_mag = sqrt(v1x*v1x + v1y*v1y)
                if (v_mag > 1e-7)
                    v1x /= v_mag
                    v1y /= v_mag
                else
                    v1x = 1.0
                    v1y = 0.0
                end
                v2x = -v1y
                v2y =  v1x

                D_xx_bayes[i,j] += (λ1.*v1x.*v1x + λ2.*v2x.*v2x)*posterior[i,j,c]
                D_xy_bayes[i,j] += (λ1.*v1x.*v1y + λ2.*v2x.*v2y)*posterior[i,j,c]
                D_yy_bayes[i,j] += (λ1.*v1y.*v1y + λ2.*v2y.*v2y)*posterior[i,j,c]
            end
        end
    end

    #ImageView.imshow(fuck)
    #return

    ProgressMeter.@showprogress for t = 1:n_iters
        weickert_matrix_diffusion!(img_dst, img_src, Δt,
                                   D_xx_bayes, D_xy_bayes, D_yy_bayes)
        @swap!(img_src, img_dst)
    end
    img_dst
end

# function pnad_test()
#     #img      = FileIO.load("../data/phantom/field2_cyst_phantom.png")
#     #img      = FileIO.load("../data/image/forearm_gray.png")
#     #img      = FileIO.load("2.png")
#     #img      = FileIO.load("../data/selections/liver/Test3_2.png")
#     img      = FileIO.load("../data/subjects/thyroid/m_KJH_000012.jpg")
#     img      = Images.Gray.(img)
#     img      = Float32.(img)
#     img_base = deepcopy(img)

#     target   = :thyroid
#     gmm      = fit_pixel_gmm(target)

#     dt        = 1.0
#     n_iters   = 50
#     img_out   = pnad(img, gmm, dt, n_iters)
#     img_out   = clamp.(img_out, 0, 1.0)
#     PNGFiles.save("ponad.png", img_out)

#     view = MosaicViews.mosaicview(ImageCore.colorview(Images.Gray, img_base),
#                                   ImageCore.colorview(Images.Gray, img_out);
#                                   nrow=1,
#                                   npad=10)
#     ImageView.imshow(view)
# end
