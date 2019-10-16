# SheafLearning.jl
This is a Julia package containing routines for solving the optimization problems in [1] 
for learning sheaf Laplacians and matrix-weighted graph Laplacians from smooth signals.
It also has the ability to compute projections onto the cone of sheaf Laplacians.

## Installing
SheafLearning.jl is not currently a registered Julia package. To install it, run the following in the Julia REPL:
```julia
using Pkg
Pkg.add("https://github.com/hansenjakob/SheafLearning.jl")
```

## Usage
Import SheafLearning.jl as usual. The main functions are `recover_sheaf_Laplacian`, `recover_mw_Laplacian`, and `project_to_sheaf_Laplacian`. 

The two sheaf recovery functions minimize the objective function 

```tr(LM) - alpha * Σ_v log(tr(L_vv)) + beta * ||L||_F^2```

where `L` is restricted to the cone of sheaf Laplacians and the cone of matrix-weighted graph Laplacians, respectively. The structure of the Laplacian is restricted by the assumption that the graph has `Nv` vertices and that vertex stalks are `dv`-dimensional. `M` here is the matrix of the outer products of the smooth signals: if columns of `X` are the signals, we have `M = XX^T`. This can also be seen (assuming mean-centered data) as a data covariance matrix.

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

Currently, this library uses a very simple interior-point method with a gradient-descent based internal loop. It is not yet particularly robust or speedy, but seems to handle Laplacians with `Nv = 100` and `dv = 2` reasonably well.


The function ```project_to_sheaf_Laplacian``` minimizes

```||L - M||_F^2```

for `L` in the cone of sheaf Laplacians over graphs with `Nv` vertices and `dv`-dimensional vertex stalks. The signature is

```julia
project_to_sheaf_Laplacian(M,Nv,dv;verbose=0)
```

This problem is translated to a conic program and solved using [SCS](https://github.com/JuliaOpt/SCS.jl).


[1] Hansen, Jakob and Ghrist, Robert. [Learning Sheaf Laplacians from Smooth Signals](https://www.math.upenn.edu/~jhansen/publications/learningsheaves.pdf). _Proceedings of the International Conference on Acoustics, Speech, and Signal Processing_, 2019.
