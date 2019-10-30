#translates a symmetric matrix M into a vector in the format required by SCS
#that is, copies the lower triangle into a vector along columns, scaling off-diagonal entries by sqrt(2)
function sym2vec(M)
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

#translates a vector from SCS format to a symmetric matrix. Requires dimension of the resulting matrix
#assumes the vector is the lower triangle, following columns
function vec2sym(V,dim)
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

#assumes the vector is the upper triangle, following columns
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

function vectorize_M_triangular(M,Nv,dv::Array{Int,1};lower=true)
    totaldims = sum(dv)
    Ne = div(Nv*(Nv-1),2)

    blockends = cumsum(dv)
    blockindices = [blockends[i]-dv[i]+1:blockends[i] for i in 1:Nv]
    edge_matrix_sizes = [(dv[i] + dv[j]) for i in 1:Nv for j in i+1:Nv]
    edge_vec_sizes = div.(edge_matrix_sizes.*(edge_matrix_sizes.+1),2)
    Me = zeros(sum(edge_vec_sizes))
    Le = zeros(size(Me))
    e = 1
    startidx = 1
    for i = 1:Nv
        for j = i+1:Nv
            indices = [blockindices[i]; blockindices[j]]
            veclength = div(edge_matrix_sizes[e]*(edge_matrix_sizes[e]+1),2)
            if lower
                Mvec = sym2vec(M[indices,indices])
                Le[startidx:startidx+veclength-1] = sym2vec(Matrix{Float64}(I,(edge_matrix_sizes[e],edge_matrix_sizes[e])))
            else
                Mvec = sym2vec_upper(M[indices,indices])
                Le[startidx:startidx+veclength-1] = sym2vec_upper(Matrix{Float64}(I,(edge_matrix_sizes[e],edge_matrix_sizes[e])))
            end
            Me[startidx:startidx+veclength-1] = Mvec
            startidx += veclength
            e += 1
        end
    end
    return Me, Le, edge_matrix_sizes
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