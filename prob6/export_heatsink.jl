using MAT
using WriteVTK

"""
    export_heatsink_vtu(x, datafile; basename=\"heatsink\")

Reconstruct full temperature field from solution `x` and `datafile` (same
format as in your MATLAB code) and write `basename.vtu` for ParaView.


Open heatsink.vtu in paraview. When the heatsink displays in grayscale,
right-click and select "color by" Temperature.
"""
function export_heatsink_vtu(x, datafile; basename="heatsink")
    # --- Load data ---
    data      = matread(datafile)
    coords    = data["coords"]              # N x 3
    tri       = Int.(data["tri"])           # M x 3 (1-based already)
    baseNodes = vec(Int.(data["baseNodes"]))

    N = size(coords, 1)

    # --- Same Gaussian base temperature as before ---
    Tinf     = 293.15   # K, ambient
    T0       = 363.15   # K, peak at center
    sigma    = 0.03     # m
    x_center = 0.0
    z_center = 0.0

    x_base = coords[baseNodes, 1]
    z_base = coords[baseNodes, 3]

    dx2   = (x_base .- x_center).^2
    dz2   = (z_base .- z_center).^2
    Tbase = Tinf .+ (T0 - Tinf) .* exp.(-(dx2 .+ dz2) ./ (2 * sigma^2))

    # Full temperature vector
    T = fill(Tinf, N)
    T[baseNodes] .= Tbase

    fixed = falses(N)
    fixed[baseNodes] .= true
    free = .!fixed
    T[free] .= x          # insert CG solution

    # --- Build VTK unstructured grid ---
    # points: 3 × N (WriteVTK expects (dim, num_points))
    points = permutedims(coords)  # from N×3 to 3×N

    # cells: vector of MeshCell{VTK_TRIANGLE}
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, tri[i, :]) 
             for i in 1:size(tri, 1)]

    # This writes basename.vtu in the current directory
    vtk_grid(basename, points, cells) do vtk
        # point data: one value per node
        vtk["Temperature"] = T
    end

    println("Wrote $(basename).vtu")
end
