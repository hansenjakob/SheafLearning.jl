### Test optimizers

#Le, loss = recover_sheaf_Laplacian(M,0,0,Nv,dv;verbose=true)
#println(norm(Le))
#println(loss)
#print(edge_matrices_to_Laplacian(Le,Nv,dv))
#@test isapprox(norm(Le),0,atol=1e-15)

Le, loss = recover_sheaf_Laplacian(M,alpha,beta,Nv,dv;verbose=true,tscale=100)

L = edge_matrices_to_Laplacian(Le,Nv,dv)

Le_vd, loss_vd = recover_sheaf_Laplacian(M,alpha,beta,Nv,[dv for i in 1:Nv];verbose=true,tscale=50)

L_vd = edge_matrices_to_Laplacian(Le_vd,Nv,[dv for i in 1:Nv])

for e = 1:length(Le_vd)
    @test Le_vd[e] ≈ Le[:,:,e]
end

@test loss_vd ≈ loss
#@test norm(L-L_vd)/norm(L) < 1e-3  

We, loss = recover_mw_Laplacian(M,alpha,beta,Nv,dv;verbose=true)