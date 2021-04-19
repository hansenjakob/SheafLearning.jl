L, dist = project_to_sheaf_Laplacian(M,Nv,dv)
@test isapprox(norm(L-M)^2,dist; atol=1e-10) 


L, dist = project_to_sheaf_Laplacian(Matrix{Float64}(I, dv*Nv, dv*Nv),Nv,dv,verbose=true)
@test L ≈ Matrix{Float64}(I,dv*Nv,dv*Nv)
@test isapprox(dist,0; atol=1e-15) 

L, dist = project_to_sheaf_Laplacian(ones(6,6),3,2,verbose=true)
@test dist > 1

if(mosek)
  L, dist = project_to_sheaf_Laplacian_mosek(M,Nv,[dv for _ in 1:Nv],verbose=true)
  @test isapprox(norm(L-M)^2,dist; atol=1e-10) 

  L, dist = project_to_sheaf_Laplacian_mosek(Matrix{Float64}(I, dv*Nv, dv*Nv),Nv,[dv for _ in 1:Nv],verbose=true)
  @test L ≈ Matrix{Float64}(I,dv*Nv,dv*Nv)
  @test isapprox(dist,0; atol=1e-15) 

  L, dist = project_to_sheaf_Laplacian_mosek(ones(6,6),3,[2,2,2],verbose=true)
  @test dist > 1
end