using SparseArrays

n = 100
e = ones(n)
A = -spdiagm( 0 => -6*e, -1 => e, 1 => e);
A = A[1:end-1, 1:end-1];
B = kron(ones(div(n,2)), [0;1]) + ones(n)



