using SheafLearning
using Test
using LinearAlgebra
using Random

rng = Random.MersenneTwister(10324)
Nv = 25 
dv = 2
Nsamples = 3000
X = randn(rng,Nv*dv,Nsamples);
M = X*X'/Nsamples;
alpha = 1.
beta = 0.02
Ne = div(Nv*(Nv-1),2) 

Le, loss = SheafLearning.recover_sheaf_Laplacian_MOI(M,alpha,beta,Nv,[dv for i in 1:Nv])


include("function_tests.jl")

include("optimizer_tests.jl")

include("scs_learning_tests.jl")

include("projection_tests.jl")

include("bounds_checking.jl")
