using SparseArrays
using Plots
using LaTeXStrings


include("CG.jl")
#include("../prob1/arnoldi_GMRES.jl")

n = 100
e = ones(n);
A = -spdiagm(0 => -6*e, -1 => e, 1 => e);
A = A[1:(end-1), 1:(end-1)];
b = kron(ones(div(n, 2)), [0; 1]) + ones(n);


# Conjugate Gradient
x_CG, residual_CG = CG(A, b, 30)
k = 1:length(residual_CG)
p = scatter(k, residual_CG,
    yscale=:log10,
    label="Method 1: CG",
    marker=:star5,
    markercolor=:black,
    title="Residual norms",
    xlabel=L"iteration $p$",
    ylabel=L"\Vert Ax - b \Vert_2",
    legend=:bottomleft)


# Generalized Minimum Residual method - GMRES

Q, H, z_GMRES, x_GMRES = arnoldi_GMRES(A, b, 30, double_GS_MV);

residual_GMRES = zeros(length(x_GMRES))

for i in 1:length(x_GMRES)
    residual_GMRES[i] = norm(A * x_GMRES[i] - b)
end

# residual_GMRES = filter(x -> x > 1e-14, residual_GMRES);
scatter!(p, k, residual_GMRES,
    label="Method 2: GMRES",
    marker=:x,
    markersize=3,
    markercolor=:blue)

# Conjugate Gradient with least sqaures
X_ls = zeros(n, 24)

for i in 1:24
    X_ls[:, i] = x_CG[i]
end

residual_CG_ls = []

for p in 1:size(X_ls, 2)
    Xp = X_ls[:, 1:p]
    M = A * Xp
    local z = (M' * M) \ (M' * b)
    x = Xp * z
    push!(residual_CG_ls, norm(A * x - b))
end

scatter!(p, 1:size(X_ls, 2), residual_CG_ls,
    label="Method 3: CG with LSQ",
    marker=:cross,
    markercolor=:red)

display(p)
savefig("prob2/prob2_residual_norms")


# zoomed plot
scatter!(p;
    yscale=:identity,
    xlims=(1.5, 2.5),
    ylims=(0.1528, 0.1532),
    title="Residual norms (zoom)")

savefig("prob2/residual_norm_zoomed")