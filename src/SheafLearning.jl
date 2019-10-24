module SheafLearning

greet() = print("SheafLearning.jl: routines for learning sheaves from data")

using LinearAlgebra
using Optim

export recover_sheaf_Laplacian, recover_mw_Laplacian
export edge_matrices_to_Laplacian, edge_weights_to_Laplacian
export project_to_sheaf_Laplacian

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
            trL[i] += tr(Le[1:dv,1:dv,e])
            trL[j] += tr(Le[dv+1:2*dv,dv+1:2*dv,e])
            e += 1
        end
    end

    obj = dot(Le,Me)
    obj += -alpha*sum(log.(trL))
    normsq = sum(Le[1:dv,dv+1:2dv,:].^2,dims=(1,2,3))
    obj += beta*normsq[1,1,1]
    if barrier
        #obj = t
        Ne = div(Nv*(Nv-1),2)
        for e = 1:Ne
            obj += -log(det(Le[:,:,e]))/t
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

# Computes the gradient for sheaf learning objective
function sheaf_obj_gradient!(grad::Array{Float64,3},Me,Le::Array{Float64,3},alpha,beta,Nv,dv,t)
    trL = zeros(Nv)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            trL[i] += tr(Le[1:dv,1:dv,e])
            trL[j] += tr(Le[dv+1:2*dv,dv+1:2*dv,e])
            e += 1
        end
    end
    trLinv = (trL).^-1
    e = 1
    grad[:,:,:] = Me[:,:,:]
    for i = 1:Nv
        for j = i+1:Nv
            grad[1:dv,dv+1:2dv,e] += beta*Le[1:dv,dv+1:2dv,e]
            grad[dv+1:2dv,1:dv,e] += beta*Le[dv+1:2dv,1:dv,e] 
            for k = 1:dv
                grad[k,k,e] += -alpha*trLinv[i]
                grad[dv+k,dv+k,e] += -alpha*trLinv[j]
            end
            grad[:,:,e] += -inv(Le[:,:,e])/t
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
        #obj *= t
        Ne = div(Nv*(Nv-1),2)
        startidx = 1
        for e = 1:Ne
            edge_matrix = reshape(Le[startidx:startidx+edge_matrix_sizes[e]^2-1],(edge_matrix_sizes[e],edge_matrix_sizes[e]))
            obj += -log(det(edge_matrix))/t
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

    grad[:] = Me[:]
    e = 1
    startidx = 1
    for i = 1:Nv
        for j = i+1:Nv
            # accumulate gradients from off-diagonal norm
            for k = 1:dv[i]
                for l = dv[i]+1:edge_matrix_sizes[e]
                    grad[startidx+edge_matrix_sizes[e]*(k-1)+l-1] += beta*Le[startidx+edge_matrix_sizes[e]*(k-1)+l-1]
                    grad[startidx+edge_matrix_sizes[e]*(l-1)+k-1] += beta*Le[startidx+edge_matrix_sizes[e]*(l-1)+k-1] 
                end
            end 

            # accumulate gradients from log trace term
            for k = 1:dv[i]
                grad[startidx+edge_matrix_sizes[e]*(k-1)+k-1] += -alpha*trLinv[i]
            end
            for k = dv[i]+1:edge_matrix_sizes[e]
                grad[startidx+edge_matrix_sizes[e]*(k-1)+k-1] += -alpha*trLinv[j]
            end 

            #accumulate gradients from barrier function
            inverse_matrix = inv(reshape(Le[startidx:startidx+edge_matrix_sizes[e]^2-1],(edge_matrix_sizes[e],edge_matrix_sizes[e])))
            grad[startidx:startidx+edge_matrix_sizes[e]^2-1] += -reshape(inverse_matrix,edge_matrix_sizes[e]^2)/t

            startidx += edge_matrix_sizes[e]^2
            e += 1
        end
    end
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
        throw(DomainError(beta,"regularization parameter alpha must be nonnegative"))
    end
end

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

"""
    interior_point(objective,gradient,initial_condition,total_constraint_degree,tol=1e-7,maxouter=20,tscale=25,verbose=false)

Runs an interior point method for convex functions with convex constraints. 
- objective: function taking (state,t,barrier) and returning objective(state) + t*barrier(state) if barrier is true and objective(state) otherwise
- gradient: function taking (grad,state,t) and mutating grad to contain the gradient of objective + t*barrier evaluated at state
- initial_condition is obvious
- total_constraint_degree is used to bound the suboptimality of the result. This is computed from the number of constraints and the degree of the barrier functions.
- tol: solution tolerance, guarantee that the suboptimality is at most tol
- maxouter: maximum number of outer iterations to take. Typically not many are required with reasonable values of tscale
- tscale: how much to scale the barrier by at each outer iteration
- verbose: print information about the progress of the algorithm
"""
function interior_point(objective,gradient,initial_condition,total_constraint_degree,tol=1e-7,maxouter=20,tscale=25,verbose=true)
    t = 1
    state = initial_condition
    m = total_constraint_degree
    oldobj = objective(state,t,false)

    # This is the main loop. At each step, minimizes a perturbed version of the objective function with a barrier function preventing the minimizer from leaving the feasible region.
    # After each loop, the barrier function is scaled, so that the sequence of minimizers follows a path that converges to the optimum of the constrained problem.
    # This is a standard, probably poorly implemented interior-point method.
    for outeriter = 1:maxouter
        gradient_data! = (param_grad,param_state) -> gradient(param_grad,param_state,t)
        obj_data = (param_state) -> objective(param_state,t,true)
        res = optimize(obj_data,gradient_data!,state,method=GradientDescent(),show_trace=verbose)
        state = Optim.minimizer(res)
        t *= tscale
        newobj = objective(state,t,false)
        if verbose
            println("step $outeriter objective: $newobj")
            println("t = $t")
        end
        if (m/t < tol)
            #if verbose
                println("Desired tolerance $tol reached")
            #end
            oldobj = newobj
            break
        end
        oldobj = newobj
    end
    return state, oldobj
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


"""
    recover_sheaf_Laplacian(M, alpha, beta, Nv, dv; tol=1e-7, maxouter=20, tscale=25, verbose=false)

Solve the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta/2 || off diagonal blocks of L ||_F^2 
s.t. L is a sheaf Laplacian for a graph with Nv vertices and vertex stalks of dimension dv

Uses a basic interior point method with a log determinant barrier, with a gradient-based optimizer from Optim.jl for each internal iteration.

Arguments
- M: data covariance matrix XX^T
- alpha, beta: regularization parameters >= 0
- Nv: number of vertices
- dv: dimension of vertex stalks
- tol: accuracy required---the interior point method guarantees this level of suboptimality
- maxouter: maximum number of iterations for the outer loop of the interior point method
- tscale: amount to scale the barrier parameter by in each iteration
- verbose: print information about the progress of the outer loop
"""
function recover_sheaf_Laplacian(M,alpha,beta,Nv,dv::Int;tol=1e-7,maxouter=20,tscale=25,verbose=false)
    check_dims(M,Nv,dv)
    check_reg_params(alpha,beta)

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

    Me, Le = vectorize_M(M,Nv,dv)

    m = Ne*2*dv
    obj_data = (param_Le,t,barrier) -> sheaf_obj(Me,param_Le,alpha,beta,Nv,dv,t,barrier)
    gradient_data! = (param_grad,param_Le,t) -> sheaf_obj_gradient!(param_grad,Me,param_Le,alpha,beta,Nv,dv,t)
    Le, oldobj = interior_point(obj_data,gradient_data!,Le,m,tol,maxouter,tscale,verbose)

    return Le, oldobj
end
"""
    recover_mw_Laplacian(M, alpha, beta, Nv, dv; tol=1e-7, maxouter=20, tscale=25, verbose=false)

Solve the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta/2 || off diagonal blocks of L ||_F^2 
s.t. L is a matrix-weighted Laplacian for a graph with Nv vertices and vertex stalks of dimension dv

Uses a basic interior point method with a log determinant barrier, with a gradient-based optimizer from Optim.jl for each internal iteration.

Arguments
- M: data covariance matrix XX^T
- alpha, beta: regularization parameters >= 0
- Nv: number of vertices
- dv: dimension of vertex stalks
- tol: accuracy required---the interior point method guarantees this level of suboptimality
- maxouter: maximum number of iterations for the outer loop of the interior point method
- tscale: amount to scale the barrier parameter by in each iteration
- verbose: print information about the progress of the outer loop
"""
function recover_mw_Laplacian(M,alpha,beta,Nv,dv::Int;tol=1e-7,maxouter=20,tscale=25,verbose=false)
    check_dims(M,Nv,dv)
    check_reg_params(alpha,beta)

    Ne = div(Nv*(Nv-1),2)
    # Take data covarance matrix and translate into a format corresponding to each edge
    Me = zeros(dv,dv,Ne)
    e = 1
    for i = 1:Nv
        for j = i+1:Nv
            Me[:,:,e] = M[dv*(i-1)+1:dv*i,dv*(i-1)+1:dv*i] + M[dv*(j-1)+1:dv*j,dv*(j-1)+1:dv*j] - 2*M[dv*(i-1)+1:dv*i,dv*(j-1)+1:dv*j]
            e += 1
        end
    end

    # Prepare initial condition
    We = zeros(dv,dv,Ne)
    for e = 1:Ne
        We[:,:,e] = Matrix{Float64}(I, dv, dv)
    end

    m = Ne*dv

    obj_data = (param_We,t,barrier) -> mw_obj(Me,param_We,alpha,beta,Nv,dv,t,barrier)
    gradient_data! = (param_grad,param_We,t) -> mw_obj_gradient!(param_grad,Me,param_We,alpha,beta,Nv,dv,t)
    We, oldobj = interior_point(obj_data,gradient_data!,We,m,tol,maxouter,tscale,verbose)

    return We, oldobj
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

function recover_sheaf_Laplacian(M,alpha,beta,Nv,dv::Array{Int,1};tol=1e-7,maxouter=20,tscale=25,verbose=false)
    check_dims(M,Nv,dv)
    check_reg_params(alpha,beta) 

    totaldims = sum(dv)
    Ne = div(Nv*(Nv-1),2)

    # Take data covarance matrix and translate into a format corresponding to each edge
    # We have to put everything in a one-dimensional vector now, which requires a lot more work
    # Initial condition setup is in this loop too.
    # blockends = cumsum(dv)
    # blockindices = [blockends[i]-dv[i]+1:blockends[i] for i in 1:Nv]
    # edge_matrix_sizes = [(dv[i] + dv[j]) for i in 1:Nv for j in i+1:Nv]
    # Me = zeros(sum(edge_matrix_sizes.^2))
    # Le = zeros(size(Me))
    # e = 1
    # startidx = 1
    # for i = 1:Nv
    #     for j = i+1:Nv
    #         indices = [blockindices[i]; blockindices[j]]
    #         veclength = (edge_matrix_sizes[e])^2
    #         Mvec = reshape(M[indices,indices],veclength)
    #         Me[startidx:startidx+veclength-1] = Mvec
    #         Le[startidx:startidx+veclength-1] = reshape(Matrix{Float64}(I,(edge_matrix_sizes[e],edge_matrix_sizes[e])),veclength)
    #         startidx += veclength
    #         e += 1
    #     end
    # end
    Me, Le, edge_matrix_sizes = vectorize_M(M,Nv,dv)

    m = sum(edge_matrix_sizes) #total degree of barrier functions 

    obj_data = (param_Le,t,barrier) -> sheaf_obj(Me,param_Le,alpha,beta,Nv,dv,edge_matrix_sizes,t,barrier)
    gradient_data! = (param_grad,param_Le,t) -> sheaf_obj_gradient!(param_grad,Me,param_Le,alpha,beta,Nv,dv,edge_matrix_sizes,t)
    Le, oldobj = interior_point(obj_data,gradient_data!,Le,m,tol,maxouter,tscale,verbose)

    Le_out = Array{Array{Float64,2},1}()
    startidx = 1
    for e = 1:Ne
        edge_matrix = reshape(Le[startidx:startidx+edge_matrix_sizes[e]^2-1],(edge_matrix_sizes[e],edge_matrix_sizes[e]))
        push!(Le_out,edge_matrix)
        startidx += edge_matrix_sizes[e]^2
    end

    return Le_out, oldobj
end


include("sheafprojection.jl")






end # module
