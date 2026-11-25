using MAT
using LinearAlgebra
using Plots
using LaTeXStrings
include("GMRES_timed.jl")

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

    time_vec = time_vec * (1e-9)
    return x_vec, time_vec
end

N = 80 # number of iterations

x_CGNE, time_CGNE = CGNE(A, b, N)
residual_CGNE = zeros(N)

_, _, _, x_GMRES, time_GMRES = GMRES_timed(A, b, N, double_GS_MV)
residual_GMRES = zeros(N)

for i in 1:N
    residual_CGNE[i] = norm(A * x_CGNE[i] - b)
    residual_GMRES[i] = norm(A * x_GMRES[i] - b)
end

k = 1:N
p = plot(k, residual_CGNE,
    yscale=:log10,
    label="CGN",
    color=:red,
    xlabel="iteration",
    ylabel=L"\Vert Ax - b \Vert_2",
    xlims=(0, 80),
    ylims=(1e-18, 1e3))

plot!(p, k, residual_GMRES,
    label="GMRES",
    line=:dash,
    color=:black)

display(p)
savefig("prob5/prob5_iterations")

t = plot(time_CGNE, residual_CGNE,
    yscale=:log10,
    label="CGN",
    color=:red,
    xlabel="CPU-time (seconds)",
    ylabel=L"\Vert Ax - b \Vert_2",
    xlims=(0, 1))

plot!(t, time_GMRES, residual_GMRES,
    label="GMRES",
    line=:dash,
    color=:black)

display(t)
savefig("prob5/prob5_timed")

