using LinearAlgebra, SCS
using SparseArrays

## Functions for translating indices between various shapes for the matrices making up sheaf Laplacians
# Translates linear indices from vectors for symmetric matrices into 2d indices for a matrix. (lower triangular index returned)
function triangle_to_square_idx(k,sqdim)
    nelements = div(sqdim*(sqdim+1),2)
    if k > nelements
        return 0,0
    end
    cok = nelements-k+1
    coj = ceil(Int,(sqrt(8*cok +1)-1)/2)
    j = sqdim - coj +1
    i = div(coj*(coj+1),2) - cok + j
    return i,j
end

# Translates (lower triangular) indices for an edge contribution matrix L_e to the corresponding block indices in the sheaf Laplacian
function edge_mat_to_block_Laplacian_idx(e,i,j,dv,Nv)
    u,v = triangle_to_square_idx(e,Nv-1)
    u += 1
    if j <= dv
        if i <= dv
            return v,v,i,j
        else
            return u,v,i-dv,j
        end
    else
        return u,u,i-dv,j-dv
    end
end

# Translates block indices in the Laplacian to regular matrix indices
function block_matrix_to_matrix_idx(u,v,i,j,dv)
    return (u-1)*dv+i,(v-1)*dv+j
end

# Translates (lower triangular) 2-d matrix indices from two-dimensional to linear indices
function square_to_triangle_idx(i,j,sqdim)
    coj = sqdim - j + 1
    cok = div(coj*(coj+1),2) - (i - j)
    k = div(sqdim*(sqdim+1),2) - cok + 1
    return k
end

# Combines the previous functions to map indices from the vectorized edge contribution matrices to the vectorized sheaf Laplacian
function edge_mat_to_Laplacian_idx(kin,Nv,dv,debug=false)
    Ne = div(Nv*(Nv-1),2)
    trianglesize = dv*(2*dv+1)
    #if kin > Ne*dv*(2*dv+1)
    #    throw DomainError(k)
    #end
    e = fld(kin-1,trianglesize) + 1
    kv = mod(kin-1,trianglesize) +1
    i,j = triangle_to_square_idx(kv,2*dv)
    if debug
        println("triangular index $kv mapped to square index ($i, $j) in edge $e")
    end
    u,v,i,j = edge_mat_to_block_Laplacian_idx(e,i,j,dv,Nv)
    if debug
        println("which mapped to block ($u, $v), indices ($i, $j)")
    end
    i,j = block_matrix_to_matrix_idx(u,v,i,j,dv)
    if debug
        println("which mapped to absolute indices ($i, $j)")
    end
    kout = square_to_triangle_idx(i,j,Nv*dv)
    if debug
        println("which mapped to triangular index $kout")
    end
    return kout
end

# Constructs the constraint matrix for the sheaf projection cone program
function build_constraint_matrix(Nv,dv)
    Ne = div(Nv*(Nv-1),2)
    ncols = Ne*dv*(2*dv+1)
    nrows = div(Nv*dv*(Nv*dv+1),2)
    sdsize = dv*(2*dv+1)
    J = [[i for i in 1:1+ncols]; [1 + i for i in 1:Ne*sdsize]]
    I = zeros(Int,1+ncols+Ne*sdsize)
    V = [1.0 for i in 1:1+ncols+Ne*sdsize] 
    I[1] = 1
    J[1] = 1 
    for k = 1:ncols
        kout = edge_mat_to_Laplacian_idx(k,Nv,dv,false)
        I[k+1] = kout+1
    end
    I[1+ncols+1:1+ncols+Ne*sdsize] = 1+nrows+1:1+nrows+Ne*sdsize 
    A = sparse(I,J,V)
    return A
end

"""
    project_to_sheaf_cone(M,Nv,dv;verbose=0)

Takes a semidefinite matrix M of size (Nv*dv)x(Nv*dv) and finds the nearest sheaf Laplacian in the Frobenius norm.
Uses SCS --- Splitting Conic Solver --- as a backend. 
Returns the Laplacian matrix L as well as the squared distance between the two matrices.

"""
function project_to_sheaf_Laplacian(M,Nv,dv;verbose=false)
    check_dims(M,Nv,dv)
    verbose_int = verbose ? 1 : 0
    Ne = div(Nv*(Nv-1),2) 
    ncols = Ne*div(dv*(dv+1),2)
    nrows = div(Nv*dv*(Nv*dv+1),2) 
    A = build_constraint_matrix(Nv,dv)
    sdsize = dv*(2dv+1) #number of SDP variables for each edge

    #linear functional for the objective
    c = zeros(ncols+1+Ne*sdsize)
    c[1] = 1
    #Vectorize M and place in the RHS vector
    b = [0; sym2vec(M); zeros(Ne*sdsize)]
    sol = SCS_solve(SCS.Direct,
    size(A)[1], 
    size(A)[2], #Dimensions of A
    -A, -b, #-Ax + s = b
    c, #minimize c'x
    0, #no equality constraints
    0, #no linear inequality constraints
    [1+nrows], #array of SOC sizes
    [2dv for i in 1:Ne], #array of SDC sizes
    0, 
    0, #no exponential constraints
    Array{Float64,1}(); #array of power cone parameters (empty)
    verbose=verbose_int)

    Le = zeros(2dv,2dv,Ne)
    k = 2
    for e = 1:Ne
        Le[:,:,e] = vec2sym(sol.x[k:k+sdsize-1],2dv)
        k += sdsize
    end
    return edge_matrices_to_Laplacian(Le,Nv,dv), sol.x[1]^2
end