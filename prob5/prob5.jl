using MAT

#include("../prob1/arnoldi_GMRES.jl")

A = matread("prob5/cgn_illustration.mat")["A"]
b = matread("prob5/cgn_illustration.mat")["b"]

function CGNE(A, b, N)
    b = vec(b)
    n = size(A, 1)
    x = zeros(n)
    r = A' * b
    p = r
    x_vec = []
    time_vec = zeros(N)
    t0 = Int(time_ns())
    for m in 1:N
        rr = r' * r
        Ap = A' * (A * p)
        alpha_m = rr / (p' * (Ap))
        x_m = x + alpha_m * p
        r_m = r - alpha_m * Ap
        beta_m = (r_m' * r_m) / rr
        p_m = r_m + beta_m * p

        x = x_m
        r = r_m
        p = p_m
        push!(x_vec, x)
        time_vec[m] = Int(time_ns())-t0
    end
    return x_vec, time_vec
end

x_CGNE, time_CGNE = CGNE(A, b, 80)
nothing

#Q_GMRES, H_GMRES, z_GMRES, x_GMRES = arnoldi_GMRES(A, b, 80, double_GS_MV)