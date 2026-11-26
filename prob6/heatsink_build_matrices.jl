using MAT
using LinearAlgebra
using Random
using Statistics
using SparseArrays
using LaTeXStrings
using Plots

Random.seed!(0)

# --- Load data ---
data = matread("heatsink_exercise_ref1.mat")

K = data["K"]          # full stiffness
F0 = data["F"]          # base load vector (without Gaussian)
coords = data["coords"]     # N x 3
tri = convert.(Int64, data["tri"])        # M x 3
baseNodes = convert.(Int64, data["baseNodes"])  # indices of base nodes (y ≈ ymin)
topNodes = convert.(Int64, data["topNodes"])   # indices of top nodes  (y ≈ ymax)

N = size(coords, 1)

# --- Parameters for Gaussian base temperature ---

Tinf = 293.15   # K, ambient
T0 = 363.15   # K, peak at center (e.g. 90°C)
sigma = 0.03     # m, Gaussian width
x_center = 0.0      # m
z_center = 0.0      # m

# Extract base node positions
x_base = coords[baseNodes, 1]
z_base = coords[baseNodes, 3]

# Gaussian temperature profile at base
dx2 = (x_base .- x_center) .^ 2
dz2 = (z_base .- z_center) .^ 2
Tbase = Tinf .+ (T0 - Tinf) .* exp.(-(dx2 .+ dz2) ./ (2 * sigma^2))

# --- Build Dirichlet-constrained system and solve ---

# Initialize full temperature vector with ambient
T = fill(Tinf, N)
T[baseNodes] .= Tbase

fixed = falses(N)
fixed[baseNodes] .= true
free = .!fixed

# Reduced system: A * Tfree = Ff - Kfc * Tfixed
A = K[free, free]
Kfc = K[free, fixed]
Ff = F0[free]

rhs = Ff - Kfc * T[fixed]

# Solve
# x = A \ rhs

# # vector mult test (part b)
# times = []
# for i in 1:10
#   test_b = randn(size(A)[2])
#   push!(times, @elapsed(A * test_b))
# end
#
# println(times)
# println(mean(times))
# println(std(times))
#
# # backslash time test (part d)
# A \ rhs # warmup
# backslash_times = [@elapsed(A \ rhs) for _ in 1:10]
#
# println("BACKSLASH")
# println(backslash_times)
# println(mean(backslash_times))
# println(std(backslash_times))



# precond test (part e)
precond = matread("heatsink_precond_ref1.mat")
T₁ = precond["T1"]
T₂ = precond["T2"]

D = spdiagm(0 => diag(A))

# norms_T1T2 = []
# times_T1T2 = []
# norms_D = []
# times_D = []
#
# warmup
# z = randn(size(A)[2])
# _ = norm(T₂\(T₁\(A*z))-z) / norm(A*z)
# _ = norm(D\(A*z) - z)/norm(A*z)
#
# for _ ∈ 1:10
#   z = randn(size(A)[2])
#   time_T1T2 = @elapsed(n_T1T2 = norm(T₂\(T₁\(A*z))-z) / norm(A*z))
#   push!(norms_T1T2, n_T1T2)
#   push!(times_T1T2, time_T1T2)
#   time_D = @elapsed(n_D = norm(D\(A*z) - z)/norm(A*z))
#   push!(norms_D, n_D)
#   push!(times_D, time_D)
# end

# println(norms_T1T2)
# println(times_T1T2)
# println(norms_D)
# println(times_D)
# println(mean(times_D))

# @show mean(norms_T1T2)
# @show mean(times_T1T2)
# @show mean(norms_D)
# @show mean(times_D)
#
# @show std(norms_T1T2)
# @show std(times_T1T2)
# @show std(norms_D)
# @show std(times_D)


include("GMRES.jl")
include("pcg.jl")

error(A, x, b) = [norm(A * x - b)/norm(b) for x in x]

m = 100

plot(1:m,
  error(A, GMRES_single_precond(A, rhs, m, double_GS_MV, D)[2], rhs),
  xlabel="Iteration",
  ylabel=L"$|| A x_m - b ||_2 / ||b||$",
  label="GMRES with precond D",
  xlims=(0, 100),
  ylims=(1e-14, 1e1),
  yscale=:log10,
  linestyle=:dash,
  linecolor=:black,
  legend=:bottomleft,
)

plot!(1:m,
  error(A, GMRES_factor_precond(A, rhs, m, double_GS_MV, T₁, T₂)[2], rhs),
  label="GMRES with precond M",
  yscale=:log10,
  linecolor=:red,
  linestyle=:solid,
)

plot!(1:m,
  pcg_v1(A, rhs, m, D, compute_residual=true)[2],
  label="PCG with precond D",
  yscale=:log10,
  linecolor=:blue,
  linestyle=:dashdot,
)

plot!(1:m,
  pcg_v2(A, rhs, m, T₁, T₂, compute_residual=true)[2],
  label="PCG with precond M",
  yscale=:log10,
  linecolor=:green,
  linestyle=:solid,
)

savefig("plot")
