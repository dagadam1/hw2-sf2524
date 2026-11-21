using Random, SparseArrays, Plots, LaTeXStrings
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
  real_x = A\b
  iter = 100
  (Q, H, z_✶, xₘ) = arnoldi_GMRES(A, b, iter, double_GS_MV)

  # x̃ = [Q[1:m, 1:m] * z_✶[m] for m in 1:iter]

  legend = if α==1
    :bottomleft
  else
    :topright
  end

  plot(1:iter,
    [norm(xm - real_x)/norm(real_x) for xm in xₘ],
    yscale=:log10,
    label=L"$||x_* - x_m||/||x_*||$",
    xlabel="Iteration",
    ylabel="Error",
    title="α=$α",
    linecolor=:red,
    legend=legend,
    legendfontsize=14,
    titlefontsize=16
  )

  plot!(1:iter,
    [norm(A * xm - b)/norm(b) for xm in xₘ],
    yscale=:log10,
    label=L"$||Ax_m - b||/||b||$",
    linestyle=:dash,
    linecolor=:blue,
  )
  savefig("gmres_alpha=$α")
end



