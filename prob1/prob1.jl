using Random, SparseArrays, Plots
include("arnoldi_GMRES.jl")

Random.seed!(0)
α_list = (1, 5, 10, 100)
n = 100

function generate_A_b(α)
  A = sprand(ComplexF64, n, n, 0.5)
  A = A + α*I
  A = A/norm(A, 1);
  b = randn(ComplexF64, n);
  return (A, b)
end

for α in α_list
  A, b = generate_A_b(α)
  real_x = A\b;
  iter = 100
  (Q, H, z_✶, xₘ) = arnoldi_GMRES(A, b, iter, double_GS_MV)

  # x̃ = [Q[1:m, 1:m] * z_✶[m] for m in 1:iter]

  plot(1:iter, [norm(xm - real_x)/norm(real_x) for xm in xₘ], yscale=:log10, label="||x_* - x_m||/||x_*||")
  savefig("gmres_rel_error_alpha=$α")

  plot(1:iter, [norm(A * xm - b)/norm(b) for xm in xₘ], yscale=:log10, label="||Ax_m - b||/||b||")
  savefig("gmres_rel_res_alpha=$α")
end



