# SheafLearning.jl
This is a Julia package containing routines for solving the optimization problems in [1] 
for learning sheaf Laplacians and matrix-weighted graph Laplacians from smooth signals.
It also has the ability to compute projections onto the cone of sheaf Laplacians. 

The package is currently largely untested and no guarantees are made regarding its abilities or performance.

## Installation
SheafLearning.jl is not currently in the General registry. To install it, run the following in the Julia REPL:
```julia
using Pkg
Pkg.add("https://github.com/hansenjakob/SheafLearning.jl")
```

## Usage
Import SheafLearning.jl as usual. The main functions are `recover_sheaf_Laplacian`, `recover_mw_Laplacian`, and `project_to_sheaf_Laplacian`. 

The two sheaf recovery functions minimize the objective 

```tr(LM) - alpha * Σ_v log(tr(L_vv)) + beta * ||offdiag(L)||_F^2```

where `L` is restricted to the cone of sheaf Laplacians and the cone of matrix-weighted graph Laplacians, respectively. The structure of the Laplacian is restricted by the assumption that the graph has `Nv` vertices and that vertex stalks are `dv`-dimensional. (`dv` may also be an array of integers of length `Nv`, giving the dimension of each stalk) `M` here is the matrix of the outer products of the smooth signals: if columns of `X` are the signals, we have `M = XX^T`. This can also be seen (assuming mean-centered data) as a data covariance matrix.

The signatures are
```julia
recover_sheaf_Laplacian(M,alpha,beta,Nv,dv,tol=1e-7,maxouter=20,tscale=25,verbose=false)
recover_mw_Laplacian(M,alpha,beta,Nv,dv,tol=1e-7,maxouter=20,tscale=25,verbose=false)
```
The functions return tuples `(Le,obj)` and `(We,obj)`, respectively, where `Le` is a 3-dimensional array whose slices are edge contribution matrices of dimension `2dv` by `2dv`, `We` is an array whose slices are edge weight matrices of dimension `dv` by `dv`, and `obj` is the objective function value at the minimizer. To convert these arrays into actual Laplacians, use the functions 
```julia
edge_matrices_to_Laplacian(Le,Nv,dv)
edge_weights_to_Laplacian(We,Nv,dv)
```

Currently, this library uses a very simple interior-point method with a gradient-descent based internal loop. It is not yet particularly robust or speedy. Another option is to use a general conic programming formulation of a nearly identical problem. For these, the functions

```julia
recover_sheaf_Laplacian_mosek(M,alpha,beta,Nv,dv; verbose=false)
recover_sheaf_Laplacian_SCS(M,alpha,beta,Nv,dv; max_iters=5000,verbose=false)
```
use the backend solvers Mosek and SCS, respectively. Mosek is a commercial package, but has free academic licenses available, while SCS is open source. The Mosek solver is currently by far the fastest and most robust.

One important note: in order to formulate the optimization problem in standard conic form, the squared frobenius norm term `beta * ||offdiag(L)||_F^2` becomes simply `beta * ||offdiag(L)||_F`, so the two approaches do not quite solve equivalent problems (unless `beta = 0`).


The function ```project_to_sheaf_Laplacian``` minimizes

```||L - M||_F^2```

for `L` in the cone of sheaf Laplacians over graphs with `Nv` vertices and `dv`-dimensional vertex stalks. The signature is

```julia
project_to_sheaf_Laplacian(M,Nv,dv;verbose=0)
```

This problem is translated to a conic program and solved using [SCS](https://github.com/JuliaOpt/SCS.jl).

More information about the functions can be found using Julia's built-in help. Type `?[function_name]` into the REPL.

## Contact
Feel free to contact me at jhansen at math.upenn.edu with questions or suggestions. More resources on cellular sheaves and their Laplacians are available at [jakobhansen.org](http://www.jakobhansen.org).



[1] Hansen, Jakob and Ghrist, Robert. [Learning Sheaf Laplacians from Smooth Signals](https://www.math.upenn.edu/~jhansen/publications/learningsheaves.pdf). _Proceedings of the International Conference on Acoustics, Speech, and Signal Processing_, 2019.
