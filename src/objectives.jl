## Objective functions, their gradients and Hessians

#Objective function for the sheaf learning problem 
function sheaf_obj(Me,Le::Array{Float64,3},alpha,beta,Nv,dv,t,barrier=true)
    #Check that the input is in the domain of the function; this ensures the line search works properly
    if !is_valid_edge_matrix(Le,1e-12)
        return Inf
    end
    trL = zeros(Nv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            for k = 1:dv
                @inbounds trL[i] += Le[k,k,e]
                @inbounds trL[j] += Le[dv+k,dv+k,e]
            end
            e += 1
        end
    end

    obj = dot(Le,Me)
    obj -= alpha*sum(log.(trL))
    @views obj += beta*norm(Le[1:dv,dv+1:2dv,:])^2 
    if barrier
        obj *= t
        Ne = div(Nv*(Nv-1),2)
        for e = 1:Ne
            @inbounds obj += -log(det(Le[:,:,e]))
        end
    end
    return obj
end

# function sheaf_obj(Me,Le::Array{Float64,3},alpha,beta,Nv,dv,t,barrier=true)
#     Ne = div(Nv*(Nv-1),2)
#     return sheaf_obj(reshape(Me,(2dv)^2*Ne),reshape(Le,(2dv)^2*Ne),alpha,beta,Nv,[dv for i in 1:Nv],[2dv for i in 1:Ne],t,barrier)
# end

#Objective function for the matrix-weighted graph learning problem
function mw_obj(Me,We::Array{Float64,3},alpha,beta,Nv,dv,t,barrier=true)
    #Check that the input is in the domain of the function; this ensures the line search works properly
    if !is_valid_edge_matrix(We,1e-12)
        return Inf
    end

    trL = zeros(Nv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            trL[i] += tr(We[:,:,e])
            trL[j] += tr(We[:,:,e]) 
            e += 1
        end
    end

    obj = dot(We,Me) #Me here is not the same as the sheaf version
    obj += -alpha*sum(log.(trL))
    obj += beta*norm(We)^2
    if barrier
        obj *= t
        Ne = div(Nv*(Nv-1),2)
        for e = 1:Ne
            obj += -log(det(We[:,:,e]))
        end
    end
    return obj
end

#Computes the gradient for sheaf learning objective
function sheaf_obj_gradient!(grad::Array{Float64,3},Me,Le::Array{Float64,3},alpha,beta,Nv,dv,t)
    trL = zeros(Nv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            for k = 1:dv
                @inbounds trL[i] += Le[k,k,e]
                @inbounds trL[j] += Le[dv+k,dv+k,e]
            end
            e += 1
        end
    end
    trL .= -alpha.*(trL).^-1
    e = 1
    grad .= Me
    @views grad[1:dv,dv+1:2dv,:] += beta.*Le[1:dv,dv+1:2dv,:]
    @views grad[dv+1:2dv,1:dv,:] += beta.*Le[dv+1:2dv,1:dv,:]
    for i = 1:Nv
        for j = i+1:Nv
            for k = 1:dv
                @inbounds grad[k,k,e] += trL[i]
                @inbounds grad[dv+k,dv+k,e] += trL[j]
            end
            grad[:,:,e] *= t
            grad[:,:,e] += -inv(Le[:,:,e])
            e += 1
        end
    end
end 

# function sheaf_obj_gradient!(grad::Array{Float64,3},Me,Le::Array{Float64,3},alpha,beta,Nv,dv,t)
#     Ne = div(Nv*(Nv-1),2)
#     grad_vec = zeros((2dv)^2*Ne)
#     sheaf_obj_gradient!(grad_vec,reshape(Me,size(grad_vec)),reshape(Le,size(grad_vec)),alpha,beta,Nv,[dv for i in 1:Nv],[2dv for i in 1:Ne],t)
#     grad[:,:,:] = reshape(grad_vec,(2dv,2dv,Ne))
# end


#Computes the gradient for matrix-weighted graph objective
function mw_obj_gradient!(grad::Array{Float64,3},Me,We::Array{Float64,3},alpha,beta,Nv,dv,t)
    trL = zeros(Nv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            trL[i] += tr(We[:,:,e])
            trL[j] += tr(We[:,:,e]) 
            e += 1
        end
    end
    trLinv = (trL).^-1
    e = 1
    grad[:,:,:] = t*Me[:,:,:]
    grad[:,:,:] += 2*t*beta*We[:,:,:]
    for i = 1:Nv
        for j = i+1:Nv
            for k = 1:dv
                grad[k,k,e] += -t*alpha*(trLinv[i]+trLinv[j])
            end
            grad[:,:,e] += -inv(We[:,:,e])
            e += 1
        end
    end
end

#Objective function for sheaf learning with stalks of varying dimension
function sheaf_obj(Me::Array{Float64,1},Le::Array{Float64,1},alpha,beta,Nv,dv::Array{Int,1},edge_matrix_sizes::Array{Int,1},t,barrier=true)
    #Check that the input is in the domain of the function; this ensures the line search works properly
    if !is_valid_edge_matrix(Le,edge_matrix_sizes,1e-12)
        return Inf
    end

    #This loop calculates the trace of each diagonal block of L as well as the squared norm of the off-diagonal blocks of L
    trL = get_trace(Le,Nv,dv,edge_matrix_sizes)
    off_diag_norm = 0.
    e = 1
    startidx = 1
    for i = 1:Nv
        for j = i+1:Nv
            #norm accumulation
            for k = 1:dv[i]
                for l = dv[i]+1:edge_matrix_sizes[e]
                    off_diag_norm += Le[startidx+edge_matrix_sizes[e]*(k-1)+l-1]^2
                end
            end
            startidx += edge_matrix_sizes[e]^2
            e += 1
        end
    end

    obj = dot(Le,Me)
    obj += -alpha*sum(log.(trL))
    obj += beta*off_diag_norm
    if barrier
        obj *= t
        Ne = div(Nv*(Nv-1),2)
        startidx = 1
        for e = 1:Ne
            edge_matrix = reshape(Le[startidx:startidx+edge_matrix_sizes[e]^2-1],(edge_matrix_sizes[e],edge_matrix_sizes[e]))
            obj += -log(det(edge_matrix))
            startidx += edge_matrix_sizes[e]^2
        end
    end
    return obj
end

function get_trace(Le::Array{Float64,1},Nv,dv::Array{Int,1},edge_matrix_sizes::Array{Int,1})
    trL = zeros(Nv)
    e = 1
    startidx = 1
    for i = 1:Nv
        for j = i+1:Nv
            #trace accumulation
            for k = 1:dv[i]
                trL[i] += Le[startidx+edge_matrix_sizes[e]*(k-1)+k-1]
            end
            for k = dv[i]+1:edge_matrix_sizes[e]
                trL[j] += Le[startidx+edge_matrix_sizes[e]*(k-1)+k-1]
            end
            startidx += edge_matrix_sizes[e]^2
            e += 1
        end
    end
    return trL
end

#Gradient for sheaf learning with stalks of varying dimension
function sheaf_obj_gradient!(grad::Array{Float64,1},Me::Array{Float64,1},Le::Array{Float64,1},alpha,beta,Nv,dv::Array{Int,1},edge_matrix_sizes::Array{Int,1},t)
    trL = get_trace(Le,Nv,dv,edge_matrix_sizes)
    trLinv = (trL).^-1

    grad[:] = t*Me[:]
    e = 1
    startidx = 1
    for i = 1:Nv
        for j = i+1:Nv
            # accumulate gradients from off-diagonal norm
            for k = 1:dv[i]
                for l = dv[i]+1:edge_matrix_sizes[e]
                    grad[startidx+edge_matrix_sizes[e]*(k-1)+l-1] += t*beta*Le[startidx+edge_matrix_sizes[e]*(k-1)+l-1]
                    grad[startidx+edge_matrix_sizes[e]*(l-1)+k-1] += t*beta*Le[startidx+edge_matrix_sizes[e]*(l-1)+k-1] 
                end
            end 

            # accumulate gradients from log trace term
            for k = 1:dv[i]
                grad[startidx+edge_matrix_sizes[e]*(k-1)+k-1] += -t*alpha*trLinv[i]
            end
            for k = dv[i]+1:edge_matrix_sizes[e]
                grad[startidx+edge_matrix_sizes[e]*(k-1)+k-1] += -t*alpha*trLinv[j]
            end 

            #accumulate gradients from barrier function
            inverse_matrix = inv(reshape(Le[startidx:startidx+edge_matrix_sizes[e]^2-1],(edge_matrix_sizes[e],edge_matrix_sizes[e])))
            grad[startidx:startidx+edge_matrix_sizes[e]^2-1] += -reshape(inverse_matrix,edge_matrix_sizes[e]^2)

            startidx += edge_matrix_sizes[e]^2
            e += 1
        end
    end
end 