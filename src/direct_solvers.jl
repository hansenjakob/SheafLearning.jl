"""
    recover_sheaf_Laplacian_direct(M, alpha, beta, Nv, dv; tol=1e-7, maxouter=20, tscale=25, verbose=false)

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
function recover_sheaf_Laplacian_direct(M,alpha,beta,Nv,dv::Int;tol=1e-7,maxouter=20,tscale=25,verbose=false)
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
    recover_mw_Laplacian_direct(M, alpha, beta, Nv, dv; tol=1e-7, maxouter=20, tscale=25, verbose=false)

Solve the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta/2 || off diagonal blocks of L ||_F^2 
s.t. L is a matrix-weighted Laplacian for a graph with Nv vertices and vertex stalks of dimension dv

Uses a basic interior point method with a log determinant barrier, with a gradient-based optimizer from Optim.jl for each internal iteration.

Arguments
- M: data covariance matrix XX^T
- alpha, beta: regularization parameters >= 0
- Nv: number of vertices
- dv: dimension of vertex stalks (cannot vary for a matrix-weighted graph)
- tol: accuracy required---the interior point method guarantees this level of suboptimality
- maxouter: maximum number of iterations for the outer loop of the interior point method
- tscale: amount to scale the barrier parameter by in each iteration
- verbose: print information about the progress of the outer loop
"""
function recover_mw_Laplacian_direct(M,alpha,beta,Nv,dv::Int;tol=1e-7,maxouter=20,tscale=25,verbose=false)
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



function recover_sheaf_Laplacian_direct(M,alpha,beta,Nv,dv::Array{Int,1};tol=1e-7,maxouter=20,tscale=25,verbose=false)
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
