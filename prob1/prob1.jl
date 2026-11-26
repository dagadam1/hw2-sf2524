using Random, SparseArrays, Plots, LaTeXStrings, LinearAlgebra
include("arnoldi_GMRES.jl")

Random.seed!(0)
α_list = (1, 5, 10, 100)
n = 100
iter = 100

function generate_A_b(α)
  A = sprand(n, n, 0.5)
  A = A + α*I
  A = A/opnorm(A, 1);
  b = randn(n);
  return (A, b)
end

θ = range(0, 2π, length=100)
circle_plot(r, x0, y0) = x0 .+ r .* cos.(θ), y0 .+ r .* sin.(θ)

single_outlier_bound(V, λ₁, m, r, x0, y0) = cond(V) * (abs(λ₁ - x0 - y0*im) + r) * abs(x0 + y0*im) / (abs(λ₁) * r) * (r/abs(x0 + y0*im)) .^ m

for α in α_list
  A, b = generate_A_b(α)
  real_x = A\b
  (Q, H, z_✶, xₘ) = arnoldi_GMRES(A, b, iter, double_GS_MV)

  # x̃ = [Q[1:m, 1:m] * z_✶[m] for m in 1:iter]

  # Localization disk. Obtained by obeservation of plots
  circle = if α == 1
    # (r, x0, y0)
    (0.1, 0.03, 0.0)
  elseif α == 5
    (0.1, 0.135, 0.0)
  elseif α == 10
    (0.09, 0.25, 0.0)
  elseif α == 100
    (0.03, 0.75, 0.0)
  end

  # Outlier eigenvalue. Obtained by obeservation of plots
  λ₁ = if α == 1
    0.75 + 0im
  elseif α == 5
    0.8 + 0im
  elseif α == 10
    0.8 + 0im
  elseif α == 100
    0.9 + 0im
  end

  legend = if α==1
    :bottomleft
  else
    :topright
  end

  eig = eigen(Matrix(A))

  plot(1:iter,
    [norm(xm - real_x)/norm(real_x) for xm in xₘ],
    yscale=:log10,
    label=L"$||x_* - x_m||/||x_*||$",
    xlabel="Iteration",
    ylabel="Error",
    title="α=$α",
    linecolor=:black,
    legend=legend,
    linestyle=:dash,
    legendfontsize=14,
    titlefontsize=16,
    xlims=(0, 100),
    ylims=(1e-17, 1e1),
  )

  plot!(1:iter,
    [norm(A * xm - b)/norm(b) for xm in xₘ],
    yscale=:log10,
    label=L"$||Ax_m - b||/||b||$",
    linecolor=:red,
  )

  plot!(1:iter,
    single_outlier_bound(eig.vectors, λ₁, 1:iter, circle...),
    yscale=:log10,
    label="theory prediction",
    linestyle=:dash,
    linecolor=:blue,
  )

  savefig("gmres_alpha=$α")

  scatter(eig.values,
    marker=:x,
    label="Eigvals",
    title="α=$α",
    ylabel="Im λ",
    xlabel="Re λ",
    color=:black,
    xlims=(-0.2, 1),
    ylims=(-0.2, 0.2),
    legend=:topleft,
  )


  plot!(circle_plot(circle...)..., color=:red, label="Localization disk")

  savefig("eigvals_alpha=$α",)
end
