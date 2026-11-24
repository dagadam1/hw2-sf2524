using LinearAlgebra


function CG(A, b, N)
    n = size(A, 1)
    x = zeros(n)
    r = b
    p = r

    x_vec = []
    residual_norms = []
    tol = 1e-140

    for m in 1:N
        alpha_m = (r' * r) / (p' * A * p)
        x_m = x + alpha_m * p
        r_m = r - alpha_m * A * p
        beta_m = (r_m' * r_m) / (r' * r)
        p_m = r_m + beta_m * p

        x = x_m
        r = r_m
        p = p_m
        push!(x_vec, x)

        residual = norm(A * x - b)
        if residual < tol
            break
        end
        push!(residual_norms, residual)

    end

    return x_vec, residual_norms
end

