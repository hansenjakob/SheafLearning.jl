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