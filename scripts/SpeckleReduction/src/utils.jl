
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
    if (mag > 0)
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

function tukey_biweight(g_norm, σ_g)
    if(g_norm < σ_g)
        r = g_norm / σ_g
        c = 1 - r*r
        c*c
    else
        0
    end
end

function logcompress(img::Array,
                     max_intensity::Real,
                     n_quant::Int,
                     dynamic_range::Real)
    x_max  = max_intensity
    y_max  = n_quant
    x_min  = 10.0.^(-dynamic_range/20)*x_max

    map(img) do x
        if (x >= x_min)
            y_max / log(x_max / x_min) * log10(x / x_min)
        else
            0.0
        end
    end
end
