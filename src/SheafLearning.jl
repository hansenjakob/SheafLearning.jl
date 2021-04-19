module SheafLearning

greet() = print("SheafLearning.jl: routines for learning sheaf Laplacians from data")

using LinearAlgebra
using Optim

export recover_sheaf_Laplacian, recover_mw_Laplacian, recover_sheaf_Laplacian_direct, recover_mw_Laplacian_direct, recover_sheaf_Laplacian_SCS, recover_sheaf_Laplacian_mosek
export edge_matrices_to_Laplacian, edge_weights_to_Laplacian
export project_to_sheaf_Laplacian, project_to_sheaf_Laplacian_mosek


include("utilities.jl")
include("sheafprojection.jl")
include("conicprogramming.jl")
include("objectives.jl")
include("optimizer.jl")
include("direct_solvers.jl")

##TODO: allow underlying graphs other than the complete graph
##TODO: add a non-convex formulation using the boundary matrix

"""
    recover_sheaf_Laplacian(M, alpha, beta, Nv, dv; backend="scs", tol=1e-7, maxouter=20, tscale=25, verbose=false)

Solve the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta/2 || off diagonal blocks of L ||_F 
s.t. L is a sheaf Laplacian for a graph with Nv vertices and vertex stalks of dimension dv

Uses one of several backends.

Arguments
- M: data covariance matrix XX^T
- alpha, beta: regularization parameters >= 0
- Nv: number of vertices
- dv: dimension of vertex stalks (either an integer or a list of length Nv)
- backend: "scs", "mosek", or "direct". the direct solver is not recommended. "mosek" requires an external Mosek installation.
These arguments are only used if backend=="direct"
- tol: accuracy required---the interior point method guarantees this level of suboptimality
- maxouter: maximum number of iterations for the outer loop of the interior point method
- tscale: amount to scale the barrier parameter by in each iteration
- verbose: print information about the progress of the outer loop
"""
function recover_sheaf_Laplacian(M, alpha, beta, Nv, dv; tol=1e-7, maxouter=20, tscale=25, verbose=false, backend="scs")
    if backend == "scs"
        if isa(dv, Int)
            dv_ = [dv for _ in 1:Nv]
        else
            dv_ = dv
        end
        return recover_sheaf_Laplacian_SCS(M, alpha, beta, Nv, dv_; verbose=verbose)
    elseif backend == "mosek"
        if isa(dv, Int)
            dv_ = [dv for _ in 1:Nv]
        else
            dv_ = dv
        end
        return recover_sheaf_Laplacian_mosek(M, alpha, beta, Nv, dv_; verbose=verbose)
    elseif backend == "direct"
        return recover_sheaf_Laplacian_direct(M, alpha, beta, Nv, dv; tol=tol, maxouter=maxouter, tscale=tscale, verbose=verbose)
    end
end

"""
    recover_mw_Laplacian(M, alpha, beta, Nv, dv; backend="direct", tol=1e-7, maxouter=20, tscale=25, verbose=false)

Solve the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta/2 || off diagonal blocks of L ||_F^2 
s.t. L is a matrix-weighted Laplacian for a graph with Nv vertices and vertex stalks of dimension dv

Will support multiple backends in the future. 

Arguments
- M: data covariance matrix XX^T
- alpha, beta: regularization parameters >= 0
- Nv: number of vertices
- dv: dimension of vertex stalks (cannot vary for a matrix-weighted graph)
- backend: currently only "direct" is supported
- tol: accuracy required---the interior point method guarantees this level of suboptimality
- maxouter: maximum number of iterations for the outer loop of the interior point method
- tscale: amount to scale the barrier parameter by in each iteration
- verbose: print information about the progress of the outer loop
"""
function recover_mw_Laplacian(M, alpha, beta, Nv, dv::Int; tol=1e-7, maxouter=20, tscale=25, verbose=false, backend="direct")
    if backend == "direct"
        return recover_mw_Laplacian_direct(M, alpha, beta, Nv, dv; tol=tol, maxouter=maxouter, tscale=tscale, verbose=verbose)
    end
end


end # module
