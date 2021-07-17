
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

    λ1 = 0.5*(A11 + A22 + tmp);
    λ2 = 0.5*(A11 + A22 - tmp);

    if (abs(λ1) > abs(λ2))
        v2x, v2y, v1x, v1y, λ2, λ1
    else
        v1x, v1y, v2x, v2y, λ1, λ2
    end 
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

function lorentzian_errornorm(x, k)
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
    y_max  = max_output_intensity
    x_min  = 10.0.^(-dynamic_range/20)*x_max

    map(img) do x
        if (x >= x_min)
            y_max / log10(x_max / x_min) * log10(x / x_min)
        else
            0.0
        end
    end
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
                                            D_yy::Array)
    M = size(img_dst, 1)
    N = size(img_dst, 2)
    a = D_xx
    b = D_xy
    c = D_yy
    for j = 1:N
        for i = 1:M
            nx = max(i - 1, 1)
            px = min(i + 1, M)
            ny = max(j - 1, 1)
            py = min(j + 1, N)

            A1 = (1/4)*(b[nx, j ] - b[i, py])
            A2 = (1/2)*(c[i,  py] + c[i, j ])
            A3 = (1/4)*(b[px, j ] + b[i, py])
            A4 = (1/2)*(a[nx, j ] + a[i, j ])
            A6 = (1/2)*(a[px, j ] + a[i, j ])
            A7 = (1/4)*(b[nx, j ] + b[i, ny])
            A8 = (1/2)*(c[i,  ny] + c[i, j ])
            A9 = (1/4)*(b[px, j ] - b[i, ny])

            img_dst[i,j] = (img_src[i,j] + Δt*(
                A1*(img_src[nx, py]) + 
                    A2*(img_src[i,  py]) + 
                    A3*(img_src[px, py]) + 
                    A4*(img_src[nx, j])  + 
                    A6*(img_src[px, j])  + 
                    A7*(img_src[nx, ny]) + 
                    A8*(img_src[i,  ny]) + 
                    A9*(img_src[px, ny])))  /
                    (1 + Δt*(A1 + A2 + A3 + A4 + A6 + A7 + A8 + A9))
        end
    end
end

# function fit_pixel_gmm(target)
#     gmm_file = if target == :liver
#         "liver_gmm.jld2"
#     else
#         "thyroid_gmm.jld2"
#     end

#     if(isfile(gmm_file))
#         FileIO.load(gmm_file, "gmm")
#     else
#         mask = if target == :liver
#             FileIO.load("convex_mask.png") 
#         else
#             FileIO.load("linear_mask.png") 
#         end
#         mask = Images.Gray.(mask) 
#         mask = Float32.(mask) .> 0

#         n_classes = if target == :liver
#             3
#         else
#             4
#         end

#         fnames = if target == :liver
#             ["../data/subjects/thyroid/Test1_1.png",
#              "../data/subjects/thyroid/Test1_2.png",
#              "../data/subjects/thyroid/Test1_3.png",
#              "../data/subjects/thyroid/Test1_4.png",
#              "../data/subjects/thyroid/Test2_1.png",
#              "../data/subjects/thyroid/Test2_2.png",
#              "../data/subjects/thyroid/Test2_3.png",
#              "../data/subjects/thyroid/Test2_4.png",
#              ]
#         else
#             ["../data/subjects/thyroid/m_min__000000.jpg",
#              "../data/subjects/thyroid/m_min__000003.jpg",
#              "../data/subjects/thyroid/m_min__000005.jpg",
#              "../data/subjects/thyroid/m_min__000007.jpg",
#              "../data/subjects/thyroid/m_min__000010.jpg",
#              "../data/subjects/thyroid/m_PJH_000000.jpg",
#              "../data/subjects/thyroid/m_PJH_000003.jpg",
#              "../data/subjects/thyroid/m_PJH_000005.jpg",
#              "../data/subjects/thyroid/m_PJH_000007.jpg",
#              "../data/subjects/thyroid/m_PJH_000010.jpg",
#              ]
#         end

#         pixels = mapreduce(vcat, fnames) do fname
#             img = FileIO.load(fname)
#             img = Float64.(Images.Gray.(img))
#             reshape(img[mask], :)
#         end
#         gmm   = GaussianMixtures.GMM(n_classes, pixels, nIter=0)
#         idx   = sortperm(gmm.μ[:,1]; rev=true)
#         gmm.μ = gmm.μ[idx,:]
#         gmm.Σ = gmm.Σ[idx,:]
#         gmm.w = gmm.w[idx]

#         GaussianMixtures.em!(gmm, reshape(pixels, (:,1)); nIter=16, varfloor=1e-5)
#         FileIO.save(gmm_file, "gmm", gmm)
#         gmm 
#     end
# end
