/*
R Code for Testing :
This R code should produce the exact same model as ML.Regression.dense.OLS_LU.
Code :

A <- matrix(c(1,0.13197,25.114,3,0.0,72.009,5,0.95613,71.9,7,0.57521,97.91,9,0.0,102.2,
11,0.23478,118.48,13,0.0,145.83,15,0.0,181.51,17,0.015403,197.38,19,0.0,214.03,
21,0.16899,216.61,23,0.64912,270.63,25,0.73172,281.17,27,0.64775,295.11,29,0.45092,314.04,
31,0.54701,331.86,33,0.29632,345.95,35,0.74469,385.31,37,0.18896,390.91,39,0.6868,423.49), nrow = 20, ncol = 3, byrow=TRUE);

Y <- A[, 3];
X1 <- A[, 1];
X2 <- A[, 2];
model <- lm(Y ~ 1 + X1 + X2);
summary(model)
*/

// Run the following code both on THOR and HTHOR
  IMPORT ML;
   value_record := RECORD
   UNSIGNED rid;
   UNSIGNED X_1;
   REAL X_2;
   REAL Y;
   END;
   d := DATASET([{1,1,0.13197,25.114},{2,3,0.0,72.009},{3,5,0.95613,71.900},{4,7,0.57521,97.906},
        {5,9,0.0,102.2},{6,11,0.23478,118.48},{7,13,0.0,145.83},{8,15,0.0,181.51},
        {9,17,0.015403,197.38},{10,19,0.0,214.03},{11,21,0.16899,216.61},{12,23,0.64912,270.63},
        {13,25,0.73172,281.17},{14,27,0.64775,295.11},{15,29,0.45092,314.04},{16,31,0.54701,331.86},
     {17,33,0.29632,345.95},{18,35,0.74469,385.31},{19,37,0.18896,390.91},{20,39,0.68678,423.49}],value_record);
   ML.ToField(d,o);
   X := O(Number = 1 OR Number = 2); // Pull out the X
	 OUTPUT(COUNT(X),NAMED('size_of_X'));
   Y := O(Number = 3); // Pull out the Y
   model_dense := ML.Regression.dense.OLS_LU(X,Y);
   OUTPUT(model_dense.Betas,NAMED('DenseModel'));

   // Make sparse X by removing 0 elements. Model produced should be the same as above.
	 X_sparse := X(value<>0);
	 OUTPUT(COUNT(X_sparse),NAMED('size_of_X_sparse'));
   model_sparse := ML.Regression.dense.OLS_LU(X_sparse,Y);
   OUTPUT(model_sparse.Betas,NAMED('SparseModel'));
	 