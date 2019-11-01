using SheafLearning
using Test
using LinearAlgebra
using Random

rng = Random.MersenneTwister(10324)
Nv = 10 
dv = 2
Nsamples = 3000
alpha = 1.
beta = 0.02
Ne = div(Nv*(Nv-1),2)

Pe = 0.2
A,Bgraph,Ne_actual = SheafLearning.er_graph(Nv,Pe,rng)
Bsheaf = SheafLearning.random_Gaussian_sheaf(Bgraph,dv,dv,rng)
L = Bsheaf'*Bsheaf

X = SheafLearning.sample_smooth_vectors_tikh(L,Nsamples,10,rng)
M = X*X'/Nsamples

include("function_tests.jl")

include("optimizer_tests.jl")

include("scs_learning_tests.jl")

X = randn(rng,Nv*dv,Nsamples);
M = X*X'/Nsamples;

include("projection_tests.jl")

include("bounds_checking.jl")
