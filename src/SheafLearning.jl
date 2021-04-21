module SheafLearning

greet() = print("SheafLearning.jl: routines for learning sheaf Laplacians from data")

using LinearAlgebra
using SparseArrays

using Optim

using MosekTools
using SCS
using MathOptInterface
const MOI = MathOptInterface

export recover_sheaf_Laplacian, recover_mw_Laplacian, recover_sheaf_Laplacian_direct, recover_mw_Laplacian_direct, recover_sheaf_Laplacian_SCS, recover_sheaf_Laplacian_mosek
export edge_matrices_to_Laplacian, edge_weights_to_Laplacian
export project_to_sheaf_Laplacian


include("utilities.jl")
include("sheafprojection.jl")
include("conicprogramming.jl")
include("objectives.jl")
include("optimizer.jl")
include("direct_solvers.jl")

##TODO: allow underlying graphs other than the complete graph
##TODO: add a non-convex formulation using the coboundary matrix

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
    recover_mw_Laplacian(M, alpha, beta, Nv, dv; backend="scs", tol=1e-7, maxouter=20, tscale=25, verbose=false)

Solve the optimization problem
min tr(LM) - alpha Σ_v log tr(L_vv) + beta/2 || off diagonal blocks of L ||_F^2 
s.t. L is a matrix-weighted Laplacian for a graph with Nv vertices and vertex stalks of dimension dv

Arguments
- M: data covariance matrix XX^T
- alpha, beta: regularization parameters >= 0
- Nv: number of vertices
- dv: dimension of vertex stalks (cannot vary for a matrix-weighted graph)
- backend: "scs", "mosek", or "direct". the direct solver is not recommended. "mosek" requires an external Mosek installation. 
- tol: accuracy required---the interior point method guarantees this level of suboptimality
- maxouter: maximum number of iterations for the outer loop of the interior point method
- tscale: amount to scale the barrier parameter by in each iteration
- verbose: print information about the progress of the outer loop
"""
function recover_mw_Laplacian(M, alpha, beta, Nv, dv::Int; tol=1e-7, maxouter=20, tscale=25, verbose=false, backend="scs")
    if backend == "scs"
        return recover_mw_Laplacian_SCS(M, alpha, beta, Nv, dv; verbose=verbose)
    elseif backend == "mosek"
        return recover_mw_Laplacian_mosek(M, alpha, beta, Nv, dv; verbose=verbose)
    elseif backend == "direct"
        return recover_mw_Laplacian_direct(M, alpha, beta, Nv, dv; tol=tol, maxouter=maxouter, tscale=tscale, verbose=verbose)
    end
end

"""
    project_to_sheaf_Laplacian(M,Nv,dv;verbose=false)

Takes a semidefinite matrix M of size (Nv*dv)x(Nv*dv) and finds the nearest sheaf Laplacian in the Frobenius norm.
backend may be either "scs" or "mosek".
Returns the Laplacian matrix L as well as the squared distance between the two matrices.

"""
function project_to_sheaf_Laplacian(M,Nv,dv;backend="scs",verbose=false)
    if backend=="scs"
        return project_to_sheaf_Laplacian_scs(M,Nv,dv;verbose=verbose)
    elseif backend=="mosek"
        return project_to_sheaf_Laplacian_mosek(M,Nv,dv;verbose=verbose)
    end
end


end # module
