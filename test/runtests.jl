using SheafLearning
using Test
using LinearAlgebra
using Random

mosek = false

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

@testset "functions and gradients" begin
include("function_tests.jl")
test_functions(M,Nv,Ne,dv,alpha,beta)
end

@testset "direct optimizers" begin
include("optimizer_tests.jl")
end

#TODO: actual tests here
@testset "conic optimizers" begin
include("conic_learning_tests.jl")
end

X = randn(rng,Nv*dv,Nsamples);
M = X*X'/Nsamples; #only need a random SPD matrix for the projection tests
@testset "projection problems" begin
include("projection_tests.jl")
end

@testset "bounds checking" begin
include("bounds_checking.jl")
end

