include("indices.jl")

#translates a symmetric matrix M into a vector in the format required by SCS
#that is, copies the lower triangle into a vector along columns, scaling off-diagonal entries by sqrt(2)
function sym2vec(M; lower=true)
    if !lower
        return sym2vec_upper(M)
    else
        dim = size(M)[1]
        Mvec = zeros(div(dim*(dim+1),2))
        k = 1
        for j = 1:dim
            Mvec[k] = M[j,j]
            k += 1
            for i = j+1:dim
                Mvec[k] = sqrt(2)*M[i,j]
                k += 1
            end
        end
        return Mvec
    end
end

#translates a vector from SCS format to a symmetric matrix. Requires dimension of the resulting matrix
#assumes the vector is the lower triangle, following columns
function vec2sym(V,dim; lower=true)
    if !lower
        return vec2sym_upper(V,dim)
    else
        M = zeros(dim,dim)
        k = 1
        for j = 1:dim
            M[j,j] = V[k]
            k += 1
            for i = j+1:dim
                v = V[k]/sqrt(2)
                M[i,j] = v
                M[j,i] = v
                k += 1
            end
        end
        return M
    end
end

#same as before, but assumes the vector is the upper triangle, following columns. Also no scaling factor.
function vec2sym_upper(V,dim)
    M = zeros(dim,dim)
    k = 1
    for j = 1:dim
        
        for i = 1:j-1
            v = V[k]#/sqrt(2)
            M[i,j] = v
            M[j,i] = v
            k += 1
        end
        M[j,j] = V[k]
        k += 1
    end
    return M
end

#converts upper triangle of M into a vector following columns
function sym2vec_upper(M)
    dim = size(M)[1]
    Mvec = zeros(div(dim*(dim+1),2))
    k = 1
    for j = 1:dim
        
        for i = 1:j-1
            #Mvec[k] = sqrt(2)*M[i,j]
            Mvec[k] = M[i,j]
            k += 1
        end
        Mvec[k] = M[j,j]
        k += 1
    end
    return Mvec
end


function vectorize_M(M,Nv,dv::Int)
    Ne = div(Nv*(Nv-1),2)
    # Take data covarance matrix and translate into a format corresponding to each edge
    Me = zeros(2dv,2dv,Ne)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            indices = [dv*(i-1)+1:dv*i; dv*(j-1)+1:dv*j]
            Me[:,:,e] = M[indices,indices]
            e += 1
        end
    end
    # Prepare initial condition
    Le = zeros(2dv,2dv,Ne)
    for e = 1:Ne
        Le[:,:,e] = Matrix{Float64}(I, dv*2, dv*2)
    end 
    return Me, Le
end


function edge_mat_sizes(Nv,dv::Array{Int,1})
   return [(dv[i] + dv[j]) for i in 1:Nv for j in i+1:Nv] 
end

function vectorize_M(M,Nv,dv::Array{Int,1})
    totaldims = sum(dv)
    Ne = div(Nv*(Nv-1),2)

    blockends = cumsum(dv)
    blockindices = [blockends[i]-dv[i]+1:blockends[i] for i in 1:Nv]
    edge_matrix_sizes = [(dv[i] + dv[j]) for i in 1:Nv for j in i+1:Nv]
    Me = zeros(sum(edge_matrix_sizes.^2))
    Le = zeros(size(Me))
    e = 1
    startidx = 1
    for i = 1:Nv
        for j = i+1:Nv
            indices = [blockindices[i]; blockindices[j]]
            veclength = (edge_matrix_sizes[e])^2
            Mvec = reshape(M[indices,indices],veclength)
            Me[startidx:startidx+veclength-1] = Mvec
            Le[startidx:startidx+veclength-1] = reshape(Matrix{Float64}(I,(edge_matrix_sizes[e],edge_matrix_sizes[e])),veclength)
            startidx += veclength
            e += 1
        end
    end
    return Me, Le, edge_matrix_sizes
end

#converts parameter data into what's required for input to the conic solvers
function vectorize_M_triangular(M,Nv,dv::Array{Int,1};lower=true)
    totaldims = sum(dv)
    Ne = div(Nv*(Nv-1),2)

    blockends = cumsum(dv)
    blockindices = [blockends[i]-dv[i]+1:blockends[i] for i in 1:Nv]
    edge_matrix_sizes = edge_mat_sizes(Nv,dv)
    edge_vec_sizes = div.(edge_matrix_sizes.*(edge_matrix_sizes.+1),2)
    Me = zeros(sum(edge_vec_sizes))
    Le = zeros(size(Me))
    e = 1
    startidx = 1
    for i = 1:Nv
        for j = i+1:Nv
            indices = [blockindices[i]; blockindices[j]]
            veclength = div(edge_matrix_sizes[e]*(edge_matrix_sizes[e]+1),2)
            Me[startidx:startidx+veclength-1] = sym2vec(M[indices,indices];lower=lower)
            Le[startidx:startidx+veclength-1] = sym2vec(Matrix{Float64}(I,(edge_matrix_sizes[e],edge_matrix_sizes[e]));lower=lower)
            startidx += veclength
            e += 1
        end
    end
    return Me, Le, edge_matrix_sizes
end

function vectorize_M_triangular_mw(M,Nv,dv::Int;lower=true)
    totaldims = Nv*dv
    Ne = div(Nv*(Nv-1),2)

    blockindices = [(i-1)*dv+1:i*dv for i in 1:Nv]
    M_differences = zeros(size(M))
    for u = 1:Nv
        for v = u+1:Nv
            M_differences[blockindices[u],blockindices[v]] = M[blockindices[u],blockindices[u]] + M[blockindices[v],blockindices[v]] - M[blockindices[u],blockindices[v]] - M[blockindices[v],blockindices[u]]
        end
    end
    edge_matrix_size = dv
    edge_vec_size = div(dv*(dv+1),2)

    Me = zeros(Ne*edge_vec_size)
    We = zeros(size(Me))
    e = 1
    startidx = 1
    for u = 1:Nv
        for v = u+1:Nv
            Me[startidx:startidx+edge_vec_size-1] = sym2vec(M_differences[blockindices[u],blockindices[v]];lower=lower)
            We[startidx:startidx+edge_vec_size-1] = sym2vec(Matrix{Float64}(I,(dv,dv));lower=lower) 
            startidx += edge_vec_size
            e += 1
        end
    end
    return Me, We
end


function is_valid_edge_matrix(Le::Array{Float64,3},eps)
    dim = size(Le)[1]
    Ne = size(Le)[3]
    for i = 1:Ne
        if det(Le[:,:,i]) < eps
            return false
        end
        for k = 1:dim
            if Le[k,k,i] < eps
                return false
            end
        end
    end
    return true
end

function is_valid_edge_matrix(Le::Array{Float64,1},blocksizes,eps)
    Ne = length(blocksizes)
    startidx = 1
    for e = 1:Ne
        edge_mat = reshape(Le[startidx:startidx+blocksizes[e]^2 - 1],(blocksizes[e],blocksizes[e]))
        if det(edge_mat) < eps
            return false
        end
        for k = 1:blocksizes[e]
            if edge_mat[k,k] < eps
                return false
            end
        end
        startidx += blocksizes[e]^2
    end
    return true
end



###User-facing utility functions
"""
    edge_matrices_to_Laplacian(Le, Nv, dv)

Compute a sheaf Laplacian L from an array Le of compressed edge contribution matrices as returned by recover_sheaf_Laplacian.

"""
function edge_matrices_to_Laplacian(Le::Array{Float64,3},Nv,dv::Int)
    L = zeros(Nv*dv,Nv*dv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            indices = [dv*(i-1)+1:dv*i; dv*(j-1)+1:dv*j]
            L[indices,indices] += Le[:,:,e]
            e += 1
        end
    end
    return L
end

function edge_matrices_to_Laplacian(Le::Array{Array{Float64,2},1},Nv,dv::Array{Int,1})
    totaldims = sum(dv)
    L = zeros(totaldims,totaldims)
    blockends = cumsum(dv)
    blockindices = [blockends[i]-dv[i]+1:blockends[i] for i in 1:Nv]
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            indices = [blockindices[i]; blockindices[j]]
            L[indices,indices] += Le[e]
            e += 1
        end
    end 
    return L
end

"""
    edge_weights_to_Laplacian(We, Nv, dv)

Compute a sheaf Laplacian L from an array We of matrix-valued edge weights as returned by recover_mw_Laplacian.

"""
function edge_weights_to_Laplacian(We,Nv,dv)
    L = zeros(Nv*dv,Nv*dv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            L[dv*(i-1)+1:dv*i,dv*(j-1)+1:dv*j] = -We[:,:,e]
            L[dv*(j-1)+1:dv*j,dv*(i-1)+1:dv*i] = -We[:,:,e]
            L[dv*(i-1)+1:dv*i,dv*(i-1)+1:dv*i] += We[:,:,e]
            L[dv*(j-1)+1:dv*j,dv*(j-1)+1:dv*j] += We[:,:,e]
            e += 1
        end
    end
    return L
end

#input validation
function check_dims(M,Nv,dv::Int)
    dims = size(M)
    if length(dims) != 2 || dims[1] != dims[2] 
        throw(DimensionMismatch("M must be a square 2D array"))
    elseif dims[1] != Nv*dv
        Msize = dims[1]
        paramsize = Nv*dv
        throw(DimensionMismatch("M has size $Msize while input of size Nv*dv = $paramsize was expected"))
    end 
end

function check_dims(M,Nv,dv::Array{Int,1})
    dims = size(M)
    if length(dv) != Nv
        throw(DimensionMismatch("dv must have Nv = $Nv entries"))
    end
    totaldims = sum(dv)
    if length(dims) != 2 || dims[1] != dims[2] 
        throw(DimensionMismatch("M must be a square 2D array"))
    elseif dims[1] != totaldims
        Msize = dims[1]
        throw(DimensionMismatch("M has size $Msize while input of size sum(dv) = $totaldims was expected"))
    end 
end

function check_reg_params(alpha,beta)
    if alpha < 0
        throw(DomainError(alpha,"regularization parameter alpha must be nonnegative"))
    end
    if beta < 0
        throw(DomainError(beta,"regularization parameter beta must be nonnegative"))
    end
end

#generators for testing. can be used to explore or benchmark.
function er_graph(Nv,Pe,rng)
    A = zeros(Nv,Nv)
    Ne = 0
    for i in 1:Nv
        for j in i+1:Nv
            if rand(rng) <= Pe
                A[i,j] = 1
                A[j,i] = 1
                Ne = Ne + 1
            end
        end
    end

    B = zeros(Ne,Nv)

    k = 1
    for i in 1:Nv
        for j in i+1:Nv
            if A[i,j] == 1
                B[k,i] = A[i,j]
                B[k,j] = -A[i,j]
                k = k + 1
            end
        end
    end
    return (A,B,Ne)
end

function random_Gaussian_sheaf(Bgraph,dv,de,rng)
    Ne, Nv = size(Bgraph)
    B = zeros(de*Ne,dv*Nv)
    for i = 1:Ne
        idx = findall(x -> x != 0, Bgraph[i,:])
        v1 = idx[1]
        v2 = idx[2]
        B[1+de*(i-1):de*i,1+dv*(v1-1):dv*v1] = Bgraph[i,v1]*randn(rng,de,dv)
        B[1+de*(i-1):de*i,1+dv*(v2-1):dv*v2] = Bgraph[i,v2]*randn(rng,de,dv)
    end

    return B
end

function sample_smooth_vectors_tikh(L,nsamples,smoothness,rng)
    n,m = size(L)
    L = L/norm(L)
    e = eigen(L)
    lambda = e.values
    V = e.vectors
    smoother = diagm((ones(n)+smoothness*lambda).^-1)
    samples = randn(rng,n,nsamples)
    samples = V*smoother*V'*samples
    for i = 1:nsamples
        samples[:,i] = samples[:,i]/norm(samples[:,i])
    end
    return samples
end