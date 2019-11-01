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

Pe = 0.2

A,Bgraph,Ne_actual = er_graph(Nv,Pe)
Bsheaf = random_Gaussian_sheaf(Bgraph,dv,dv)
L = Bsheaf'*Bsheaf

X = sample_smooth_vectors_tikh(L,Nsamples,10)
M = X*X'/Nsamples

Le, loss = recover_sheaf_Laplacian_mosek(M,alpha,beta,Nv,[dv for i in 1:Nv];verbose=true)

