#TODO: write conic programs for connection Laplacians

"""
    recover_sheaf_Laplacian_mosek(M,alpha,beta,Nv,dv; verbose=false)    

Solves the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta || offdiag(L) ||_F 
with L constrained to the cone of sheaf Laplacians.
Note that this problem is slightly different than the one solved by other implementations:
the Frobenius norm term is not squared. This makes it simpler to implement in standard conic optimization solvers.

Sets up the optimization problem and calls Mosek via MathOptInterface, hence requires Mosek to be installed.

dv should be an array of length Nv, containing the vertex stalk dimensions.
"""
function recover_sheaf_Laplacian_mosek(M,alpha,beta,Nv,dv::Array{Int,1}; verbose=false)
    check_dims(M,Nv,dv)
    check_reg_params(alpha,beta) 
    optimizer = MOI.Bridges.full_bridge_optimizer(Mosek.Optimizer(QUIET = !verbose),Float64)
    
    model = MOI.Utilities.Model{Float64}()
    cache_opt = MOI.Utilities.CachingOptimizer(model,optimizer)
    x = construct_sheaf_problem_mosek(cache_opt,M,alpha,beta,Nv,dv;verbose=verbose)
    
    MOI.optimize!(cache_opt)
    status = MOI.get(optimizer, MOI.TerminationStatus())
    if verbose
        println("Optimizer terminated with status ",status)
    end
    sol = MOI.get(optimizer, MOI.VariablePrimal(), x)
    Ne = div(Nv*(Nv-1),2)

    edge_matrix_sizes = edge_mat_sizes(Nv,dv) 
    edge_vec_sizes = div.(edge_matrix_sizes.*(edge_matrix_sizes.+1),2)
    
    Le = Array{Array{Float64,2},1}()
    startidx = 1
    for e = 1:Ne
        push!(Le,vec2sym_upper(sol[startidx:startidx+edge_vec_sizes[e]-1],edge_matrix_sizes[e]))
        startidx += edge_vec_sizes[e]
    end
    return Le

end


# This function takes an empty model and instantiates the correctly parameterized sheaf optimization problem as a MOI object
# TODO: figure out how to use this to efficiently resolve with different parameters
function construct_sheaf_problem_mosek(optimizer,M,alpha,beta,Nv,dv; verbose=false)
    totaldims = sum(dv)
    Ne = div(Nv*(Nv-1),2)

    Mevec, Levec, edge_matrix_sizes = vectorize_M_triangular(M,Nv,dv;lower=false)
    SOC_block_sizes = [dv[i]*dv[j] for i in 1:Nv for j in i+1:Nv]
    total_SOC_size = sum([dv[i]*dv[j] for i in 1:Nv for j in i+1:Nv])

    
    obj_coefficients = [Mevec; beta; [-alpha for i in 1:Nv];  zeros(2Nv); zeros(total_SOC_size)] #coefficients for linear objective
    edge_vec_sizes = div.(edge_matrix_sizes.*(edge_matrix_sizes.+1),2)
    total_sdc_vector_size = sum(edge_vec_sizes)

    x = MOI.add_variables(optimizer,length(obj_coefficients)) #variables for the model
    objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(obj_coefficients,x),0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),objective) #add objective function to model
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE) #we're minimizing the objective
    
    # Semidefinite constraints for the edge matrices
    if verbose
        println("Constructing and adding SDP constraints")
    end
    startidx = 0
    for e = 1:Ne
        Le = MOI.VectorOfVariables(x[startidx+1:startidx+edge_vec_sizes[e]])
        MOI.add_constraint(optimizer,Le,MOI.PositiveSemidefiniteConeTriangle(edge_matrix_sizes[e]))
        startidx += edge_vec_sizes[e]
    end

    # SOC constraints for the off-diagonal norm term
    # These consist of a single conic constraint together with equality constraints to extract the off-diagonal blocks from the SDP variables
    # MOI apparently does not allow putting other conic constraints directly on semidefinite variables
    if verbose
        println("Constructing and adding SOC constraints")
    end
    
    start_soc_indices = total_sdc_vector_size + 1 + 3Nv
    SOC_indices_list = [total_sdc_vector_size + 1; start_soc_indices+1:start_soc_indices+total_SOC_size]
    
    k = 1
    startidx = 0
    e = 1
    #this loop goes one scalar entry at a time
    for v = 1:Nv
        for u = v+1:Nv
            for i = dv[v]+1:dv[v]+dv[u]
                for j = 1:dv[v]
                    col_idx = startidx + square_to_triangle_idx_upper(i,j,dv[v]+dv[u])
                    term = MOI.ScalarAffineTerm.([1.; -1.], [x[col_idx]; x[SOC_indices_list[k+1]]])
                    MOI.add_constraint(optimizer,MOI.ScalarAffineFunction(term,0.0),MOI.EqualTo(0.0))
                    k += 1
                end
            end
            startidx += edge_vec_sizes[e]
            e += 1
        end
    end
    SOC_cone = MOI.VectorOfVariables(x[SOC_indices_list])
    MOI.add_constraint(optimizer,SOC_cone,MOI.SecondOrderCone(total_SOC_size + 1))
    
    if verbose
        println("Constructing and adding exponential cone constraints")
    end
    
    #Equality constraints for the log(trL_vv) terms. There's one variable z_v for each vertex v, constrained here to be equal to tr(L_vv)

    #This loop extracts the indices of the semidefinite variables that contribute to each tr(L_vv) term.
    trL_indices = [Array{Int64,1}() for v in 1:Nv]
    startidx = 0
    e = 1
    for v = 1:Nv
        for u = v+1:Nv
            for i = 1:dv[v]
                col_idx = startidx + square_to_triangle_idx_upper(i,i,dv[v]+dv[u])
                push!(trL_indices[v], col_idx)
            end
            for i = dv[v]+1:dv[v]+dv[u]
                col_idx = startidx + square_to_triangle_idx_upper(i,i,dv[v]+dv[u])
                push!(trL_indices[u],col_idx)
            end
            startidx += edge_vec_sizes[e]
            e += 1
        end
    end
    
    for v = 1:Nv
        constraint_term = MOI.ScalarAffineTerm.([ones(length(trL_indices[v])); -1.], [x[trL_indices[v]]; x[total_sdc_vector_size + 1 + 2Nv + v]])
        MOI.add_constraint(optimizer,MOI.ScalarAffineFunction(constraint_term,0.0), MOI.EqualTo(0.0)) #z_v == tr(L_vv)
        MOI.add_constraint(optimizer,MOI.SingleVariable(x[total_sdc_vector_size + 1 + Nv + v]),MOI.EqualTo(1.0)) #y_v == 1
    end
    
    # exponential cone constraints. Each cone has three variables (x_v,y_v,z_v). We previously constrained z_v == tr(L_vv) and y_v == 1, to make the constraint equivalent to x_v <= log tr(L_vv)
    startidx = total_sdc_vector_size + 1
    for v = 1:Nv
        trLv_cone_indices = [startidx + v ; startidx + Nv + v ; startidx + 2Nv + v]
        trLv_cone = MOI.VectorOfVariables(x[trLv_cone_indices])
        MOI.add_constraint(optimizer,trLv_cone,MOI.ExponentialCone()) #exp(x_v) <= z_v
    end
    return x
end

"""
    recover_mw_Laplacian_mosek(M,alpha,beta,Nv,dv; verbose=false)    

Solves the optimization problem
min tr(LM) + alpha Σ_v log tr(L_vv) + beta || offdiag(L) ||_F 
with L constrained to the cone of matrix-weighted Laplacians.
Note that this problem is slightly different than the one solved by other implementations:
the Frobenius norm term is not squared. This makes it simpler to implement in standard conic optimization solvers.

Sets up the optimization problem and calls Mosek via MathOptInterface, hence requires Mosek to be installed.

dv should be an integer.
"""
function recover_mw_Laplacian_mosek(M,alpha,beta,Nv,dv::Int; verbose=false)
    check_dims(M,Nv,dv)
    check_reg_params(alpha,beta) 
    optimizer = MOI.Bridges.full_bridge_optimizer(Mosek.Optimizer(QUIET = !verbose),Float64)
    
    model = MOI.Utilities.Model{Float64}()
    cache_opt = MOI.Utilities.CachingOptimizer(model,optimizer)
    x = construct_mw_problem_mosek(cache_opt,M,alpha,beta,Nv,dv;verbose=verbose)
    
    MOI.optimize!(cache_opt)
    status = MOI.get(optimizer, MOI.TerminationStatus())
    if verbose
        println("Optimizer terminated with status ",status)
    end
    sol = MOI.get(optimizer, MOI.VariablePrimal(), x)
    Ne = div(Nv*(Nv-1),2)

    # Me, Levec, edge_matrix_sizes = vectorize_M_triangular(M,Nv,dv;lower=false)
    edge_vec_size = div(dv*(dv+1),2)
    
    We = Array{Array{Float64,2},1}()
    startidx = 1
    for e = 1:Ne
        push!(We,vec2sym_upper(sol[startidx:startidx+edge_vec_size-1],dv))
        startidx += edge_vec_size
    end
    return We

end

# This function takes an empty model and instantiates the correctly parameterized sheaf optimization problem as a MOI object
function construct_mw_problem_mosek(optimizer,M,alpha,beta,Nv,dv; verbose=false)
    totaldims = Nv*dv
    Ne = div(Nv*(Nv-1),2)

    Mevec, Levec = vectorize_M_triangular_mw(M,Nv,dv;lower=false)
    # SOC_block_sizes = [dv[i]*dv[j] for i in 1:Nv for j in i+1:Nv]
    # total_SOC_size = sum([dv[i]*dv[j] for i in 1:Nv for j in i+1:Nv])
    SOC_block_size = div(dv*(dv+1),2) 
    total_SOC_size = Ne*SOC_block_size

    
    obj_coefficients = [Mevec; beta; [-alpha for i in 1:Nv];  zeros(2Nv); zeros(total_SOC_size)] #coefficients for linear objective
    edge_vec_size = div(dv*(dv+1),2)
    edge_matrix_size = dv
    total_sdc_vector_size = Ne*edge_vec_size 

    x = MOI.add_variables(optimizer,length(obj_coefficients)) #variables for the model
    objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(obj_coefficients,x),0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),objective) #add objective function to model
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE) #we're minimizing the objective
    
    # Semidefinite constraints for the edge matrices
    if verbose
        println("Constructing and adding SDP constraints")
    end
    startidx = 0
    for e = 1:Ne
        Le = MOI.VectorOfVariables(x[startidx+1:startidx+edge_vec_size])
        MOI.add_constraint(optimizer,Le,MOI.PositiveSemidefiniteConeTriangle(edge_matrix_size))
        startidx += edge_vec_size
    end

    # SOC constraints for the off-diagonal norm term
    # These consist of a single conic constraint together with equality constraints to extract the off-diagonal blocks from the SDP variables
    # MOI apparently does not allow putting other conic constraints directly on semidefinite variables
    if verbose
        println("Constructing and adding SOC constraints")
    end
    
    start_soc_indices = total_sdc_vector_size + 1 + 3Nv
    SOC_indices_list = [total_sdc_vector_size + 1; start_soc_indices+1:start_soc_indices+total_SOC_size]
    
    k = 1
    startidx = 0
    e = 1
    #this loop goes one scalar entry at a time
    for v = 1:Nv
        for u = v+1:Nv
            for i = 1:dv
                for j = i:dv
                    col_idx = startidx + square_to_triangle_idx_upper(i,j,dv)
                    term = MOI.ScalarAffineTerm.([1.; -1.], [x[col_idx]; x[SOC_indices_list[k+1]]])
                    MOI.add_constraint(optimizer,MOI.ScalarAffineFunction(term,0.0),MOI.EqualTo(0.0))
                    k += 1
                end
            end
            startidx += edge_vec_size
            e += 1
        end
    end

    SOC_variables = MOI.VectorOfVariables(x[SOC_indices_list])
    MOI.add_constraint(optimizer,SOC_variables,MOI.SecondOrderCone(total_SOC_size + 1))
    
    if verbose
        println("Constructing and adding exponential cone constraints")
    end
    
    #Equality constraints for the log(trL_vv) terms. There's one variable z_v for each vertex v, constrained here to be equal to tr(L_vv)

    #This loop extracts the indices of the semidefinite variables that contribute to each tr(L_vv) term.
    trL_indices = [Array{Int64,1}() for v in 1:Nv]
    startidx = 0
    e = 1
    for v = 1:Nv
        for u = v+1:Nv
            for i = 1:dv
                col_idx = startidx + square_to_triangle_idx_upper(i,i,dv)
                push!(trL_indices[v], col_idx)
                push!(trL_indices[u], col_idx)
            end
            startidx += edge_vec_size
            e += 1
        end
    end
    
    for v = 1:Nv
        constraint_term = MOI.ScalarAffineTerm.([ones(length(trL_indices[v])); -1.], [x[trL_indices[v]]; x[total_sdc_vector_size + 1 + 2Nv + v]])
        MOI.add_constraint(optimizer,MOI.ScalarAffineFunction(constraint_term,0.0), MOI.EqualTo(0.0)) #z_v == tr(L_vv)
        MOI.add_constraint(optimizer,MOI.SingleVariable(x[total_sdc_vector_size + 1 + Nv + v]),MOI.EqualTo(1.0)) #y_v == 1
    end
    
    # exponential cone constraints. Each cone has three variables (x_v,y_v,z_v). We previously constrained z_v == tr(L_vv) and y_v == 1, to make the constraint equivalent to x_v <= log tr(L_vv)
    startidx = total_sdc_vector_size + 1
    for v = 1:Nv
        trLv_cone_indices = [startidx + v ; startidx + Nv + v ; startidx + 2Nv + v]
        trLv_cone = MOI.VectorOfVariables(x[trLv_cone_indices])
        MOI.add_constraint(optimizer,trLv_cone,MOI.ExponentialCone()) #exp(x_v) <= z_v
    end
    return x
end

"""
    recover_sheaf_Laplacian_SCS(M,alpha,beta,Nv,dv,verbose=false)

Solves the optimization problem
min tr(LM) + alpha Σ_v log tr(L_vv) + beta || offdiag(L) ||_F 
with L constrained to the cone of sheaf Laplacians.
Note that this problem is slightly different than the one solved by other implementations:
the Frobenius norm term is not squared. This makes it easier to implement in standard conic optimization solvers.

Sets up the problem and calls SCS to solve.

dv should be an array of length Nv, containing the vertex stalk dimensions.
"""
function recover_sheaf_Laplacian_SCS(M,alpha,beta,Nv,dv::Array{Int,1};max_iters=5000,verbose=false)
    #unfortunately, we can't use the MOI interface because it is broken.
    check_dims(M,Nv,dv)
    check_reg_params(alpha,beta) 
    totaldims = sum(dv)
    Ne = div(Nv*(Nv-1),2)

    verbose_int = verbose ? 1 : 0
    Me, Levec, edge_matrix_sizes = vectorize_M_triangular(M,Nv,dv)
    c = [Me; [-alpha for i in 1:Nv]; beta]
    edge_vec_sizes = div.(edge_matrix_sizes.*(edge_matrix_sizes.+1),2)
    total_sdc_vector_size = sum(edge_vec_sizes)
    SDC_coefficients = spzeros(total_sdc_vector_size,length(c))
    for i = 1:total_sdc_vector_size
        SDC_coefficients[i,i] = 1
    end

    total_SOC_size = sum([dv[i]*dv[j] for i in 1:Nv for j in i+1:Nv])
    SOC_coefficients = spzeros(total_SOC_size+1,length(c))
    SOC_coefficients[1,length(c)] = 1
    k = 2
    startidx = 0
    e = 1
    for v = 1:Nv
        for u = v+1:Nv
            for i = dv[v]+1:dv[v]+dv[u]
                for j = 1:dv[v]
                    col_idx = startidx + square_to_triangle_idx(i,j,dv[v]+dv[u]) 
                    SOC_coefficients[k,col_idx] = 1
                    k += 1
                end
            end
        startidx += edge_vec_sizes[e]
        e += 1
        end
    end

    EXP_coefficients = spzeros(3*Nv,length(c))
    EXP_rhs = zeros(3*Nv)
    k = 1
    e = 1
    startidx = 0
    for v = 1:Nv
        for u = v+1:Nv
            for i = 1:dv[v]
                col_idx = startidx + square_to_triangle_idx(i,i,dv[v]+dv[u])
                EXP_coefficients[3*v,col_idx] += 1
            end
            for i = dv[v]+1:dv[v]+dv[u]
                col_idx = startidx + square_to_triangle_idx(i,i,dv[v]+dv[u])
                EXP_coefficients[3*u,col_idx] += 1
            end
            startidx += edge_vec_sizes[e]
            e += 1
        end
        EXP_coefficients[3*(v-1)+1,total_sdc_vector_size+v] = 1
        EXP_rhs[3*(v-1)+2] = 1
    end

    constraint_matrix = [-SOC_coefficients; -SDC_coefficients; -EXP_coefficients]
    rhs_constraint = [zeros(total_SOC_size+1); zeros(total_sdc_vector_size); EXP_rhs]

    mn = size(constraint_matrix)
    sol = SCS_solve(SCS.Direct,
                mn[1],
                mn[2],
                constraint_matrix,
                rhs_constraint,
                c,
                0, #No equality constraints
                0, #no linear constraints
                [size(SOC_coefficients)[1]], #array of SOC cone sizes
                #1, #number of SOC constraints
                edge_matrix_sizes, #array of SDC cone sizes
                #Ne, #number of SDC constraints
                Nv, #number of exponential cone constraints
                0, #number of dual exponential cone constraints
                Array{Float64,1}(), #array of power cone parameters
                #0 #number of power cone constraints
                verbose = verbose_int,
                max_iters = max_iters
                )

    #extract the array of edge matrices
    Le = Array{Array{Float64,2},1}()
    startidx = 1
    for e = 1:Ne
        push!(Le,vec2sym(sol.x[startidx:startidx+edge_vec_sizes[e]-1],edge_matrix_sizes[e]))
        startidx += edge_vec_sizes[e]
    end
    return Le
end

"""
    recover_mw_Laplacian_SCS(M,alpha,beta,Nv,dv,verbose=false)

Solves the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta || offdiag(L) ||_F 
with L constrained to the cone of matrix-weighted Laplacians.
Note that this problem is slightly different than the one solved by other implementations:
the Frobenius norm term is not squared. This makes it easier to implement in standard conic optimization solvers.

Sets up the problem and calls SCS to solve.

dv should be an integer with the constant vertex stalk dimension. 
"""
function recover_mw_Laplacian_SCS(M,alpha,beta,Nv,dv::Int;max_iters=5000,verbose=false)
    check_dims(M,Nv,dv)
    check_reg_params(alpha,beta) 
    totaldims = Nv*dv 
    Ne = div(Nv*(Nv-1),2)

    verbose_int = verbose ? 1 : 0
    Me, Levec = vectorize_M_triangular_mw(M,Nv,dv) 
    edge_matrix_size = dv
    edge_matrix_sizes = [edge_matrix_size for _ in 1:Ne]

    c = [Me; [-alpha for i in 1:Nv]; beta] #coefficients for the linear objective function
    edge_vec_size = div(dv*(dv+1),2) 
    total_sdc_vector_size = Ne*edge_vec_size 

    #construct semidefinite constraint matrix
    SDC_coefficients = spzeros(total_sdc_vector_size,length(c))
    for i = 1:total_sdc_vector_size
        SDC_coefficients[i,i] = 1 
    end

    #construct second-order cone constraint matrix
    total_SOC_size = div(dv*(dv+1),2) * div(Nv*(Nv-1),2) 
    SOC_coefficients = spzeros(total_SOC_size+1,length(c))
    SOC_coefficients[1,length(c)] = 1
    k = 2
    startidx = 0
    e = 1
    for v = 1:Nv
        for u = v+1:Nv
            for i = 1:dv
                col_idx = startidx + square_to_triangle_idx(i,i,dv)
                k += 1
                for j = i+1:dv
                    col_idx = startidx + square_to_triangle_idx(i,j,dv) 
                    SOC_coefficients[k,col_idx] = 2
                    k += 1
                end
            end
        startidx += edge_vec_size
        e += 1
        end
    end

    #construct exponential cone constraint matrix
    EXP_coefficients = spzeros(3*Nv,length(c))
    EXP_rhs = zeros(3*Nv)
    k = 1
    e = 1
    startidx = 0
    for v = 1:Nv
        for u = v+1:Nv
            for i = 1:dv
                col_idx = startidx + square_to_triangle_idx(i,i,dv)
                EXP_coefficients[3*v,col_idx] += 1
            end
            startidx += edge_vec_size
            e += 1
        end
        EXP_coefficients[3*(v-1)+1,total_sdc_vector_size+v] = 1
        EXP_rhs[3*(v-1)+2] = 1
    end

    constraint_matrix = [-SOC_coefficients; -SDC_coefficients; -EXP_coefficients]
    rhs_constraint = [zeros(total_SOC_size+1); zeros(total_sdc_vector_size); EXP_rhs]

    mn = size(constraint_matrix)
    sol = SCS_solve(SCS.Direct,
                mn[1],
                mn[2],
                constraint_matrix,
                rhs_constraint,
                c,
                0, #No equality constraints
                0, #no linear constraints
                [size(SOC_coefficients)[1]], #array of SOC cone sizes
                #1, #number of SOC constraints
                edge_matrix_sizes, #array of SDC cone sizes
                #Ne, #number of SDC constraints
                Nv, #number of exponential cone constraints
                0, #number of dual exponential cone constraints
                Array{Float64,1}(), #array of power cone parameters
                #0 #number of power cone constraints
                verbose = verbose_int,
                max_iters = max_iters
                )

    #extract the array of edge matrices
    Le = Array{Array{Float64,2},1}()
    startidx = 1
    for e = 1:Ne
        push!(Le,vec2sym(sol.x[startidx:startidx+edge_vec_size-1],edge_matrix_size))
        startidx += edge_vec_size
    end
    return Le

end