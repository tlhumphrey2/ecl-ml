/*
R-Code : The following R code should give the same model has ML.Classify.Logistic does.
"""""""""""""""
B <- matrix(c(1,35.0,149.0,0,2,11.0,138.0,0,3,12.0,148.0,1,4,16.0,156.0,0,
              5,32.0,152.0,0,6,16.0,157.0,0,7,14.0,165.0,0,8,8.0,152.0,1,
              9,35.0,177.0,0,10,0.0,158.0,1,11,40.0,166.0,0,12,28.0,165.0,0,
              13,23.0,0.0,0,14,52.0,178.0,1,15,46.0,0.0,0,16,29.0,0.0,1,
              17,30.0,172.0,0,18,21.0,163.0,0,19,21.0,164.0,0,20,20.0,189.0,1,
              21,34.0,0.0,1,22,43.0,184.0,1,23,0.0,174.0,1,24,39.0,177.0,1,
              25,0.0,0.0,1,26,0.0,175.0,1,27,32.0,173.0,1,28,0.0,173.0,1,
              29,20.0,162.0,0,30,25.0,0.0,1,31,22.0,173.0,1,32,25.0,171.0,1),nrow = 32, ncol = 4, byrow=TRUE);

Y <- B[, 4];
X1 <- B[, 2];
X2 <- B[, 3];

model <- glm(Y ~X1+X2, family="binomial");
summary(model)
*/ 

// Run the following code both on THOR and HTHOR
   IMPORT ML;

   value_record := RECORD
         UNSIGNED   rid;
         REAL     age;
         REAL     height;
         integer1   sex; // 0 = female, 1 = male
   END;

   d := DATASET([{1,35.0,149.0,0},{2,11.0,138.0,0},{3,12.0,148.0,1},{4,16.0,156.0,0},
                 {5,32.0,152.0,0},{6,16.0,157.0,0},{7,14.0,165.0,0},{8,8.0,152.0,1},
                 {9,35.0,177.0,0},{10,0.0,158.0,1},{11,40.0,166.0,0},{12,28.0,165.0,0},
                 {13,23.0,0.0,0},{14,52.0,178.0,1},{15,46.0,0.0,0},{16,29.0,0.0,1},
                 {17,30.0,172.0,0},{18,21.0,163.0,0},{19,21.0,164.0,0},{20,20.0,189.0,1},
                 {21,34.0,0.0,1},{22,43.0,184.0,1},{23,0.0,174.0,1},{24,39.0,177.0,1},
                 {25,0.0,0.0,1},{26,0.0,175.0,1},{27,32.0,173.0,1},{28,0.0,173.0,1},
                 {29,20.0,162.0,0},{30,25.0,0.0,1},{31,22.0,173.0,1},{32,25.0,171.0,1}
								 ]
                 ,value_record);

   ML.ToField(d,flds0);
   flds := ML.Discretize.ByRounding(flds0);
	 X := flds0(number<=2);
	 Y := flds(number=3);
   LogReg := ML.Classify.Logistic(0.0);
   ModelD := LogReg.LearnC(X,Y);
   OUTPUT(LogReg.Model(ModelD), NAMED('DenseModel'));

   // Make sparse X by removing zeros. Resulting model should be the same as above.
   X_sparse := X(value<>0);
   ModelS := LogReg.LearnC(X_sparse,Y);
   OUTPUT(LogReg.Model(ModelS), NAMED('SparseModel'));