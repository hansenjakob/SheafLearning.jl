using SheafLearning
Nv = 20
dv = 2
Nsamples = 3000
X = randn(Nv*dv,Nsamples);
M = X*X'/Nsamples;
alpha = 1
beta = 0.02


Le, loss = recover_sheaf_Laplacian(M,alpha,beta,Nv,dv)

