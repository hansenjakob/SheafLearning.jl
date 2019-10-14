using SheafLearning
using Test

Nv = 10
dv = 2
Nsamples = 3000
X = randn(Nv*dv,Nsamples);
M = X*X'/Nsamples;
alpha = 1
beta = 0.02


Le, loss = recover_sheaf_Laplacian(M,alpha,beta,Nv,dv,true)

We, loss = recover_mw_Laplacian(M,alpha,beta,Nv,dv,true)

Le = project_to_sheaf_Laplacian(M,Nv,dv;verbose=1)

@test_throws DimensionMismatch recover_sheaf_Laplacian([1 1],alpha,beta,Nv,dv)
@test_throws DimensionMismatch recover_sheaf_Laplacian([1 2; 3 4],alpha,beta,2,3)
@test_throws DimensionMismatch recover_mw_Laplacian([1 1],alpha,beta,Nv,dv)
@test_throws DimensionMismatch recover_mw_Laplacian([1 2; 3 4],alpha,beta,2,3)
@test_throws DimensionMismatch project_to_sheaf_Laplacian([1 1],Nv,dv)
@test_throws DimensionMismatch project_to_sheaf_Laplacian([1 2; 3 4],2,3)