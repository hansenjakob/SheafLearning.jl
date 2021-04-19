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
