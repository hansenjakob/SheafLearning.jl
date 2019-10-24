import Pkg
Pkg.activate("..")
using SheafLearning
using Test
using LinearAlgebra
using Random

rng = Random.MersenneTwister(10324)
Nv = 10
dv = 2
Nsamples = 3000
X = randn(rng,Nv*dv,Nsamples);
M = X*X'/Nsamples;
alpha = 1
beta = 0.02

Le, loss = recover_sheaf_Laplacian(M,alpha,beta,Nv,[dv for i in 1:Nv];verbose=true)
