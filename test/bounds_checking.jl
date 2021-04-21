@test_throws DimensionMismatch recover_sheaf_Laplacian([1 1],alpha,beta,Nv,dv)
@test_throws DimensionMismatch recover_sheaf_Laplacian([1 2; 3 4],alpha,beta,2,3)
@test_throws DimensionMismatch SheafLearning.recover_sheaf_Laplacian_mosek([1 1],alpha,beta,Nv,[dv for i in 1:Nv])
@test_throws DimensionMismatch SheafLearning.recover_sheaf_Laplacian_mosek([1 2; 3 4],alpha,beta,2,[3, 3])
@test_throws DimensionMismatch SheafLearning.recover_sheaf_Laplacian_SCS([1 1],alpha,beta,Nv,[dv for i in 1:Nv])
@test_throws DimensionMismatch SheafLearning.recover_sheaf_Laplacian_SCS([1 2; 3 4],alpha,beta,2,[3, 3])
@test_throws DimensionMismatch recover_mw_Laplacian([1 1],alpha,beta,Nv,dv)
@test_throws DimensionMismatch recover_mw_Laplacian([1 2; 3 4],alpha,beta,2,3)
@test_throws DimensionMismatch project_to_sheaf_Laplacian([1 1],Nv,dv)
@test_throws DimensionMismatch project_to_sheaf_Laplacian([1 2; 3 4],2,3)