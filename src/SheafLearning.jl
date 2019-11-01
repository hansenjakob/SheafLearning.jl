module SheafLearning

greet() = print("SheafLearning.jl: routines for learning sheaves from data")

using LinearAlgebra
using Optim

export recover_sheaf_Laplacian, recover_mw_Laplacian, recover_sheaf_Laplacian_SCS, recover_sheaf_Laplacian_mosek
export edge_matrices_to_Laplacian, edge_weights_to_Laplacian
export project_to_sheaf_Laplacian


include("utilities.jl")
include("sheafprojection.jl")
include("conicprogramming.jl")
include("objectives.jl")



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
    
    state = initial_condition
    m = total_constraint_degree
    oldobj = objective(state,1,false)
    t = 1 #m/oldobj

    # This is the main loop. At each step, minimizes a perturbed version of the objective function with a barrier function preventing the minimizer from leaving the feasible region.
    # After each loop, the barrier function is scaled, so that the sequence of minimizers follows a path that converges to the optimum of the constrained problem.
    # This is a standard, probably poorly implemented interior-point method.
    for outeriter = 1:maxouter
        gradient_data! = (param_grad,param_state) -> gradient(param_grad,param_state,t)
        obj_data = (param_state) -> objective(param_state,t,true)
        res = optimize(obj_data,gradient_data!,state,method=LBFGS(),show_trace=verbose,show_every=20)
        state = Optim.minimizer(res)
        t *= tscale
        newobj = objective(state,t,false)
        if verbose
            println("step ", outeriter, " objective: ", newobj)
            println("t = ", t)
        end
        if (m/t < tol)
            if verbose
                println("Desired tolerance ", tol, " reached with duality gap bounded by ", m/t)
            end
            oldobj = newobj
            break
        end
        oldobj = newobj
    end
    return state, oldobj
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
- dv: dimension of vertex stalks (either an integer or a list of length Nv)
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

end # module
