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


Le, loss = recover_sheaf_Laplacian(M,alpha,beta,Nv,dv,true)

L = edge_matrices_to_Laplacian(Le,Nv,dv)

Le, loss = recover_sheaf_Laplacian(M,alpha,beta,Nv,[dv for i in 1:Nv],true)

L = edge_matrices_to_Laplacian(Le,Nv,[dv for i in 1:Nv])

We, loss = recover_mw_Laplacian(M,alpha,beta,Nv,dv,true)

L, dist = project_to_sheaf_Laplacian(M,Nv,dv)

@test isapprox(norm(L-M)^2,dist; atol=1e-10) 

L, dist = project_to_sheaf_Laplacian(Matrix{Float64}(I, dv*Nv, dv*Nv),Nv,dv)
@test L ≈ Matrix{Float64}(I,dv*Nv,dv*Nv)
@test isapprox(dist,0; atol=1e-15) 

L, dist = project_to_sheaf_Laplacian(ones(6,6),3,2)

@test dist > 1

@test_throws DimensionMismatch recover_sheaf_Laplacian([1 1],alpha,beta,Nv,dv)
@test_throws DimensionMismatch recover_sheaf_Laplacian([1 2; 3 4],alpha,beta,2,3)
@test_throws DimensionMismatch recover_mw_Laplacian([1 1],alpha,beta,Nv,dv)
@test_throws DimensionMismatch recover_mw_Laplacian([1 2; 3 4],alpha,beta,2,3)
@test_throws DimensionMismatch project_to_sheaf_Laplacian([1 1],Nv,dv)
@test_throws DimensionMismatch project_to_sheaf_Laplacian([1 2; 3 4],2,3)
