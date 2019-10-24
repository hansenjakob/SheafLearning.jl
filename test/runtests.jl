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


SheafLearning.recover_sheaf_Laplacian_SCS(M,10,0,Nv,[dv for i in 1:Nv];verbose=true)

### Test objective functions 
#@test SheafLearning.sheaf_obj(ones(2dv,2dv,Ne),ones(2dv,2dv,Ne),0,0,Nv,dv,1,false) ≈ norm(ones(2dv,2dv,Ne))^2

Me, Le = SheafLearning.vectorize_M(M,Nv,dv)
for e = 1:Ne
    A = randn(rng,2dv,2dv)
    Le[:,:,e] = A'*A
end

obj_const = SheafLearning.sheaf_obj(Me,Le,alpha,beta,Nv,dv,1,true) 

dv_vec = [dv for i in 1:Nv]
Me_vec, Le_vec, edge_matrix_sizes = SheafLearning.vectorize_M(M,Nv,dv_vec)
Le_vec = reshape(Le,size(Me_vec))
obj_vec = SheafLearning.sheaf_obj(Me_vec,Le_vec,alpha,beta,Nv,dv_vec,edge_matrix_sizes,1,true)

@test obj_const == obj_vec

### Test gradient functions
grad = zeros(size(Le))
grad_vec = zeros(size(Le_vec))

SheafLearning.sheaf_obj_gradient!(grad,Me,Le,alpha,beta,Nv,dv,1)
SheafLearning.sheaf_obj_gradient!(grad_vec,Me_vec,Le_vec,alpha,beta,Nv,dv_vec,edge_matrix_sizes,1)

@test reshape(grad,(2dv)^2*Ne) == grad_vec

SheafLearning.sheaf_obj_gradient!(grad,Me,Le./10000,alpha,beta,Nv,dv,500000)
SheafLearning.sheaf_obj_gradient!(grad_vec,Me_vec,Le_vec./10000,alpha,beta,Nv,dv_vec,edge_matrix_sizes,500000)
println(norm(reshape(grad,(2dv)^2*Ne)-grad_vec))
@test reshape(grad,(2dv)^2*Ne) ≈ grad_vec

### Test optimizers

#Le, loss = recover_sheaf_Laplacian(M,0,0,Nv,dv;verbose=true)
#println(norm(Le))
#println(loss)
#print(edge_matrices_to_Laplacian(Le,Nv,dv))
#@test isapprox(norm(Le),0,atol=1e-15)



Le, loss = recover_sheaf_Laplacian(M,alpha,beta,Nv,dv;verbose=true)

L = edge_matrices_to_Laplacian(Le,Nv,dv)

Le_vd, loss_vd = recover_sheaf_Laplacian(M,alpha,beta,Nv,[dv for i in 1:Nv];verbose=true)

L_vd = edge_matrices_to_Laplacian(Le_vd,Nv,[dv for i in 1:Nv])

for e = 1:length(Le_vd)
    @test Le_vd[e] ≈ Le[:,:,e]
end

@test loss_vd ≈ loss
#@test norm(L-L_vd)/norm(L) < 1e-3  

We, loss = recover_mw_Laplacian(M,alpha,beta,Nv,dv;verbose=true)

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
