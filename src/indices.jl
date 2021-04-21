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

function square_to_triangle_idx_upper(i,j,sqdim)
    veclength = div(sqdim*(sqdim+1),2)
    coi = sqdim - i + 1
    coj = sqdim - j + 1
    cok = square_to_triangle_idx(coi,coj,sqdim)
    k = veclength - cok + 1
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
