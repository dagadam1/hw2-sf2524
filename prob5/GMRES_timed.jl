using LinearAlgebra
function GMRES_timed(A, b, m, gs_func)
    # [Q,H]=arnoldi(A,b,m)
    # A simple implementation of the Arnoldi method.
    # The algorithm will return an Arnoldi "factorization":
    #   Q*H(1:m+1,1:m)-A*Q(:,1:m)=0
    # where Q is an orthogonal basis of the Krylov subspace
    # and H a Hessenberg matrix.
    #
    # Example:
    #  A=randn(100); b=randn(100,1);
    #  m=10;
    #  [Q,H]=arnoldi(A,b,m);
    #  should_be_zero1=norm(Q*H-A*Q(:,1:m))
    #  should_be_zero2=norm(Q'*Q-eye(m+1))
    b = vec(b)
    hw1_good_gs = gs_func
    n=length(b)
    e₁norm_b = zeros(n+1)
    e₁norm_b[1] = norm(b)
    Q=ComplexF64.(zeros(n, m+1))
    H=ComplexF64.(zeros(m+1, m))
    Q[:, 1]=b/norm(b)
    z_✶ = []
    xₘ = []

    time_vec = zeros(m)
    t0 = Int(time_ns())

    for k=1:m
        w=A*Q[:, k] # Matrix-vector product 
        # w: n x 1
        # with last element        
        ### Orthogonalize w against columns of Q
        # replace this with a orthogonalization 
        (h, beta, worth)=hw1_good_gs(Q, w, k)
        ### Put Gram-Schmidt coefficients into H
        H[1:(k+1), k].=[h; beta]
        ### normalize
        Q[:, k+1].=worth/beta
        # Least squares
        push!(z_✶, (H[1:(k+1), 1:k]' * H[1:(k+1), 1:k]) \ (H[1:(k+1), 1:k]' * e₁norm_b[1:(k+1)]))
        push!(xₘ, Q[:, 1:k] * z_✶[end])

        time_vec[k] = Int(time_ns())-t0
    end

    time_vec = time_vec * (1e-9)
    return (Q, H, z_✶, xₘ, time_vec)
end

function single_GS_MV(Q, w, k)
    # Single Classical Gram-Schmidt matrix multiplication version
    # Q: basis matrix
    # w: "new" vector
    # k: "end" of Q
    # h: Gram-Schmidt coefficients

    h = (@views Q[:, 1:k]')*w
    z=w-Q[:, 1:k]*h
    β=norm(z)

    # z: residual
    # β: norm of z
    # q: new column of Q
    return (h, β, z)
end

function single_GS_for(Q, w, k)
    # Single Classical Gram-Schmidt for-loop version 
    h = zeros(k)
    z = w
    for i in 1:k
        h[i] = (@views Q[:, i]')*w
        z = z-h[i]*Q[:, i]
    end
    β = norm(z)
    return (h, β, z)
end

function modified_GS_for(Q, w, k)
    # Modifed Gram-Schmidt for-loop version 
    h = zeros(k)
    z = w
    for i in 1:k
        h[i] = (@views Q[:, i]')*z
        z = z-h[i]*Q[:, i]
    end
    β = norm(z)
    return (h, β, z)
end

function double_GS_MV(Q, w, k)
    # Double Classical Gram-Schmidt matrix multiplication version

    h = (@views Q[:, 1:k]')*w
    z = w-Q[:, 1:k]*h
    g = (@views Q[:, 1:k]')*z
    z = z-Q[:, 1:k]*g

    h = h+g
    β=norm(z)
    return (h, β, z)
end

function triple_GS_MV(Q, w, k)
    # Triple Classical Gram-Schmidt matrix multiplication version

    h = (@views Q[:, 1:k]')*w
    z = w-Q[:, 1:k]*h
    g = (@views Q[:, 1:k]')*z
    z = z-Q[:, 1:k]*g
    g2 = (@views Q[:, 1:k]')*z
    z = z-Q[:, 1:k]*g2

    h = h+g+g2
    β=norm(z)
    return (h, β, z)
end