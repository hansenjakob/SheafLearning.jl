using SheafLearning
using LinearAlgebra
using Random

rng = Random.MersenneTwister(70345)
alpha = 1.
beta = 0.02

dv = 2

for Nv = [10, 25, 50, 100, 150]
    for dv = 1:2
        Nsamples = 5*Nv^2
        Ne = div(Nv*(Nv-1),2)
        Pe = 1.1*log(Nv)/Nv

        A,Bgraph,Ne_actual = SheafLearning.er_graph(Nv,Pe,rng)
        Bsheaf = SheafLearning.random_Gaussian_sheaf(Bgraph,dv,dv,rng)

        L = Bsheaf'*Bsheaf
        X = SheafLearning.sample_smooth_vectors_tikh(L,Nsamples,10,rng)
        M = X*X'/Nsamples

        println("Nv = ", Nv, ", backend = SCS")
        @time SheafLearning.recover_sheaf_Laplacian_SCS(M,alpha,beta,Nv,[dv for i in 1:Nv];verbose=false)

        println("Nv = ", Nv, ", backend = Mosek")
        @time SheafLearning.recover_sheaf_Laplacian_mosek(M,alpha,beta,Nv,[dv for i in 1:Nv];verbose=false)
    end
end


