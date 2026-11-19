using SparseArrays
using Plots
using LaTeXStrings


include("CG.jl")
include("../prob1/arnoldi_GMRES.jl")

n = 100
e = ones(n);
A = -spdiagm( 0 => -6*e, -1 => e, 1 => e);
A = A[1:end-1, 1:end-1];
b = kron(ones(div(n,2)), [0;1]) + ones(n);


# Conjugate Gradient
x_CG, residual_CG = CG(A, b)

k = 1:length(residual_CG) 
p = scatter(k, residual_CG,
    yscale=:log10,
    label=CG,
    title="Residual norms",
    xlabel= L"iteration $p$",
    ylabel = L"\Vert Ax - b \Vert_2")

# Generalized Minimum Residual method
Q, H, z, x_GMRES = arnoldi_GRMS(A, b, 100, double_GS_MV)

residual_GMRES = norm(A .* x - b)

