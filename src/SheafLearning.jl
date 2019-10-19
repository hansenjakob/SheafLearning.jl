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
    obj += beta*norm(Le[1:dv,dv+1:dv*2,:])^2
    if barrier
        obj *= t
        Ne = div(Nv*(Nv-1),2)
        for e = 1:Ne
            obj += -log(det(Le[:,:,e]))
        end
    end
    return obj
end

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
            trL[i] += tr(Le[1:dv,1:dv,e])
            trL[j] += tr(Le[dv+1:2*dv,dv+1:2*dv,e])
            e += 1
        end
    end
    trLinv = (trL).^-1
    e = 1
    grad[:,:,:] = t*Me[:,:,:]
    for i = 1:Nv
        for j = i+1:Nv
            grad[1:dv,dv+1:2dv,e] += t*beta*Le[1:dv,dv+1:2dv,e]
            grad[dv+1:2dv,1:dv,e] += t*beta*Le[dv+1:2dv,1:dv,e] 
            for k = 1:dv
                grad[k,k,e] += -t*alpha*trLinv[i]
                grad[dv+k,dv+k,e] += -t*alpha*trLinv[j]
            end
            grad[:,:,e] += -inv(Le[:,:,e])
            e += 1
        end
    end
end 

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


function is_valid_edge_matrix(Le::Array{Float64,3},eps)
    dim = size(Le)[1]
    Ne = size(Le)[3]
    for i = 1:Ne
        if det(Le[:,:,i]) <= eps
            return false
        end
        for k = 1:dim
            if Le[k,k,i] <= eps
                return false
            end
        end
    end
    return true
end

function check_dims(M,Nv,dv)
    dims = size(M)
    if length(dims) != 2 || dims[1] != dims[2] 
        throw(DimensionMismatch("M must be a square 2D array"))
    elseif dims[1] != Nv*dv
        Msize = dims[1]
        paramsize = Nv*dv
        throw(DimensionMismatch("M has size $Msize while input of size Nv*dv = $paramsize was expected"))
    end 
end

"""
    edge_matrices_to_Laplacian(Le, Nv, dv)

Compute a sheaf Laplacian L from an array Le of compressed edge contribution matrices as returned by recover_sheaf_Laplacian.

"""
function edge_matrices_to_Laplacian(Le,Nv,dv)
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
function interior_point(objective,gradient,initial_condition,total_constraint_degree,tol=1e-7,maxouter=20,tscale=25,verbose=false)
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
        res = optimize(obj_data,gradient_data!,state,LBFGS())
        state = Optim.minimizer(res)
        t *= tscale
        newobj = objective(state,t,false)
        if verbose
            println("step $outeriter objective: $newobj")
            println("t = $t")
        end
        if (m/t < tol)
            if verbose
                println("Desired tolerance $tol reached")
            end
            oldobj = newobj
            break
        end
        oldobj = newobj
    end
    return state, oldobj
end


"""
    recover_sheaf_Laplacian(M, alpha, beta, Nv, dv, tol=1e-7, maxouter=20, tscale=25, verbose=false)

Solve the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta || off diagonal blocks of L ||_F^2 
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
function recover_sheaf_Laplacian(M,alpha,beta,Nv,dv,tol=1e-7,maxouter=20,tscale=25,verbose=false)
    check_dims(M,Nv,dv)

    if alpha < 0 || beta < 0
        # throw a BoundsError
    end

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

    m = Ne*2*dv
    obj_data = (param_Le,t,barrier) -> sheaf_obj(Me,param_Le,alpha,beta,Nv,dv,t,barrier)
    gradient_data! = (param_grad,param_Le,t) -> sheaf_obj_gradient!(param_grad,Me,param_Le,alpha,beta,Nv,dv,t)
    Le, oldobj = interior_point(obj_data,gradient_data!,Le,m,tol,maxouter,tscale,verbose)

    return Le, oldobj
end
"""
    recover_mw_Laplacian(M, alpha, beta, Nv, dv, tol=1e-7, maxouter=20, tscale=25, verbose=false)

Solve the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta || off diagonal blocks of L ||_F^2 
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
function recover_mw_Laplacian(M,alpha,beta,Nv,dv,tol=1e-7,maxouter=20,tscale=25,verbose=false)
    check_dims(M,Nv,dv)

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

include("sheafprojection.jl")






end # module
