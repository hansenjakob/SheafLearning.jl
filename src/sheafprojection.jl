using LinearAlgebra, SCS
using SparseArrays
import MathOptInterface 
using MosekTools
const MOI = MathOptInterface

##TODO: implement a Mosek backend for the projection problem

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
    project_to_sheaf_Laplacian(M,Nv,dv;verbose=0)

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

"""
    project_to_sheaf_Laplacian_mosek(M,Nv,dv;verbose=0)

Takes a semidefinite matrix M of size (Nv*sum(dv))x(Nv*sum(dv)) and finds the nearest sheaf Laplacian in the Frobenius norm.
Uses Mosek as a backend. 
Returns the Laplacian matrix L as well as the squared distance between the two matrices.

"""
function project_to_sheaf_Laplacian_mosek(M,Nv,dv;verbose=false)
    check_dims(M,Nv,dv)

    optimizer = MOI.Bridges.full_bridge_optimizer(Mosek.Optimizer(QUIET = !verbose),Float64)
    
    model = MOI.Utilities.Model{Float64}()
    cache_opt = MOI.Utilities.CachingOptimizer(model,optimizer)

    totaldims = sum(dv)
    Ne = div(Nv*(Nv-1),2)

    Me, Levec, edge_matrix_sizes = vectorize_M(M,Nv,dv)
    Mvec = vec(M)
    edge_vec_sizes = edge_matrix_sizes.*edge_matrix_sizes
    total_sdc_vector_size = sum(edge_vec_sizes)
    block_starts = cumsum(dv) .- dv[1] 

    c = [zeros(length(Levec)); 1; zeros(totaldims^2)]
    x = MOI.add_variables(cache_opt,length(c)) #variables for the model
    # x[1:length(Levec)] is the vectorized Le
    # x[length(Levec)+1] is the t coordinate of the SOC
    # x[length(Levec)+2:end] is the residual vec(L - M)

    objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c,x),0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),objective) #add objective function to model
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE) #we're minimizing the objective

    if verbose
        println("Constructing and adding SDP constraints")
    end
    startidx = 0
    for e = 1:Ne
        Le = MOI.VectorOfVariables(x[startidx+1:startidx+edge_vec_sizes[e]])
        MOI.add_constraint(cache_opt,Le,MOI.PositiveSemidefiniteConeSquare(edge_matrix_sizes[e]))
        startidx += edge_vec_sizes[e]
    end


    #This loop finds which indices in vec(Le) contribute to each index in vec(L).
    #It works by iterating through pairs of vertices (u,v) and then indices (i,j) in the edge matrix for the edge u \sim v.
    #Depending on which block of the edge matrix the indices fall in, we get different output indices in vec(L).
    if verbose
        println("Constructing and adding linear equality constraints")
    end
    map_indices = [Array{Int64,1}() for k in 1:totaldims^2]
    k = 1
    vec_idx = (i,j,vdim) -> (j-1)*vdim + 1 + i - 1
    for u = 1:Nv
        for v = u+1:Nv
            for j = 1:dv[u]
                Lj = block_starts[u] + j 
                for i = 1:dv[u]
                    Li = block_starts[u] + i
                    kout = vec_idx(Li,Lj,totaldims) 
                    push!(map_indices[kout],k)
                    k += 1
                end
                for i = 1:dv[v]
                    Li = block_starts[v] + i
                    kout = vec_idx(Li,Lj,totaldims) 
                    push!(map_indices[kout],k)
                    k += 1
                end
            end
            for j = 1:dv[v]
                Lj = block_starts[v] + j 
                for i = 1:dv[u]
                    Li = block_starts[u] + i
                    kout = vec_idx(Li,Lj,totaldims) 
                    push!(map_indices[kout],k)
                    k += 1
                end
                for i = 1:dv[v]
                    Li = block_starts[v] + i
                    kout = vec_idx(Li,Lj,totaldims) 
                    push!(map_indices[kout],k)
                    k += 1
                end
            end
        end
    end

    #Actually add the equality constraints to the model
    for k = 1:totaldims^2
        term = MOI.ScalarAffineTerm.([ones(length(map_indices[k])); 1.], [x[map_indices[k]]; x[length(Levec)+1+k]])
        MOI.add_constraint(cache_opt,MOI.ScalarAffineFunction(term,0.0),MOI.EqualTo(Mvec[k]))
    end

    if verbose
        println("Adding second order cone constraint")
    end
    SOC_variables = MOI.VectorOfVariables(x[length(Levec)+1:length(Levec)+1+totaldims^2])
    MOI.add_constraint(cache_opt,SOC_variables,MOI.SecondOrderCone(1+totaldims^2))

    MOI.optimize!(cache_opt)
    status = MOI.get(optimizer, MOI.TerminationStatus())
    if verbose
        println("Optimizer terminated with status ",status)
    end
    sol = MOI.get(optimizer, MOI.VariablePrimal(), x)
    optval = MOI.get(optimizer,MOI.ObjectiveValue())

    Le = Array{Array{Float64,2},1}()
    startidx = 1
    for e = 1:Ne
        push!(Le,reshape(sol[startidx:startidx+edge_vec_sizes[e]-1],(edge_matrix_sizes[e],edge_matrix_sizes[e])))
        startidx += edge_vec_sizes[e]
    end
    L = edge_matrices_to_Laplacian(Le,Nv,dv)

    return L, optval^2


end