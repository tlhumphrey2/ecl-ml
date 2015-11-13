 IMPORT ML;
IMPORT * FROM ML;
IMPORT * FROM ML.Mat;
IMPORT * FROM ML.Types;
IMPORT * FROM $;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

/*
  Logistic Regression implementation base on the iteratively-reweighted least squares (IRLS) algorithm:
  http://www.cs.cmu.edu/~ggordon/IRLS-example

  Logistic Regression module parameters:
  - Ridge: an optional ridge term used to ensure existance of Inv(X'*X) even if 
    some independent variables X are linearly dependent. In other words the Ridge parameter
    ensures that the matrix X'*X+mRidge is non-singular.
  - Epsilon: an optional parameter used to test convergence
  - MaxIter: an optional parameter that defines a maximum number of iterations

  The inputs to the Logis module are:
  a) A training dataset X of discretized independant variables
  b) A dataset of class results Y.

*/

  EXPORT Logistic_sparse(REAL8 Ridge=0.00001, REAL8 Epsilon=0.000000001, UNSIGNED2 MaxIter=200) := MODULE(Logistic_Model)
  
    Logis(DATASET(Types.NumericField) X,DATASET(Types.NumericField) Y) := MODULE
      SHARED mu_comp := ENUM ( Beta = 1,  Y = 2, VC = 3, Err = 4 );
      SHARED RebaseY := Utils.RebaseNumericField(Y);
      SHARED Y_Map := RebaseY.Mapping(1);
      Y_0 := RebaseY.ToNew(Y_Map);
      mY := Types.ToMatrix(Y_0);
      mX_0 := Types.ToMatrix(X);
      mX := IF(NOT EXISTS(mX_0), 
                    Mat.Vec.ToCol(Mat.Vec.From(Mat.Has(mY).Stats.xmax, 1.0), 1), 
                    Mat.InsertColumn(mX_0, 1, 1.0)); // Insert X1=1 column

      mXstats := Mat.Has(mX).Stats;
      mX_n := mXstats.XMax;
      mX_m := mXstats.YMax;

      mW := Mat.Vec.ToCol(Mat.Vec.From(mX_n,1.0),1);
      mRidge := Mat.Vec.ToDiag(Mat.Vec.From(mX_m,ridge));
      mBeta0 := Mat.Vec.ToCol(Mat.Vec.From(mX_m,0.0),1);  
      mBeta00 := Mat.MU.To(mBeta0, mu_comp.Beta);
      OldExpY_0 := Mat.Vec.ToCol(Mat.Vec.From(mX_n,-1.0),1); // -ones(size(mY))
      OldExpY_00 := Mat.MU.To(OldExpY_0, mu_comp.Y);
      mInv_xTWx0 := Mat.Identity(mX_m);
      mInv_xTWx00 := Mat.MU.To(mInv_xTWx0, mu_comp.VC);
      

      Step(DATASET(Mat.Types.MUElement) BetaPlusY) := FUNCTION
        OldExpY := Mat.MU.From(BetaPlusY, mu_comp.Y);
        AdjY := Mat.Mul(mX, Mat.MU.From(BetaPlusY, mu_comp.Beta));
        // expy =  1./(1+exp(-adjy))
        ExpY := Mat.Each.Reciprocal(Mat.Each.Add(Mat.Each.Exp(Mat.Scale(AdjY, -1)),1));
        // deriv := expy .* (1-expy)
        Deriv := Mat.Each.Mul(expy,Mat.Each.Add(Mat.Scale(ExpY, -1),1));
        // wadjy := w .* (deriv .* adjy + (y-expy))
        W_AdjY := Mat.Each.Mul(mW,Mat.Add(Mat.Each.Mul(Deriv,AdjY),Mat.Sub(mY, ExpY)));
        // weights := spdiags(deriv .* w, 0, n, n)
        Weights := Mat.Vec.ToDiag(Mat.Vec.FromCol(Mat.Each.Mul(Deriv, mW),1));
        // Inv_xTWx := Inv(x' * weights * x + mRidge)
        Inv_xTWx := Mat.Inv(Mat.Add(Mat.Mul(Mat.Mul(Mat.Trans(mX), weights), mX), mRidge));
        // mBeta := Inv_xTWx * x' * wadjy
        mBeta :=  Mat.Mul(Mat.Mul(Inv_xTWx, Mat.Trans(mX)), W_AdjY);
        err := SUM(Mat.Each.Abs(Mat.Sub(ExpY, OldExpY)),value); 
        mErr := DATASET([{1,1,err}], Mat.Types.Element);
        RETURN Mat.MU.To(mBeta, mu_comp.Beta)+
              Mat.MU.To(ExpY, mu_comp.Y)+
              Mat.MU.To(Inv_xTWx, mu_comp.VC)+
              Mat.MU.To(mErr, mu_comp.Err);
      END;

      MaxErr := mX_n*Epsilon;
      IErr := Mat.MU.To(DATASET([{1,1,mX_n*Epsilon + 1}], Mat.Types.Element), mu_comp.Err);
      SHARED BetaPair := LOOP(mBeta00+OldExpY_00+mInv_xTWx00+IErr, (COUNTER <= MaxIter) 
                  AND (Mat.MU.From(ROWS(LEFT), mu_comp.Err)[1].value > MaxErr), Step(ROWS(LEFT)));  
      BetaM := Mat.MU.From(BetaPair, mu_comp.Beta);
      rebasedBetaNF := RebaseY.ToOld(Types.FromMatrix(BetaM), Y_Map);
      BetaNF := Types.FromMatrix(Mat.Trans(Types.ToMatrix(rebasedBetaNF)));
      // convert Beta into NumericField dataset, and shift Number down by one to ensure the intercept Beta0 has id=0
      EXPORT Beta := PROJECT(BetaNF,TRANSFORM(Types.NumericField,SELF.Number := LEFT.Number-1;SELF:=LEFT;));
      
      varCovar := Mat.MU.From(BetaPair, mu_comp.VC);
      SEM := Mat.Vec.ToCol(Mat.Vec.FromDiag(varCovar), 1);
      rebasedSENF := RebaseY.ToOld(Types.FromMatrix(SEM), Y_map);
      SENF := Types.FromMatrix(Mat.Trans(Types.ToMatrix(rebasedSENF)));
      EXPORT SE := PROJECT(SENF,TRANSFORM(Types.NumericField,SELF.Number := LEFT.Number-1;SELF:=LEFT;));
      
      Res0 := PROJECT(Beta, TRANSFORM(l_model, SELF.Id := COUNTER+Base,SELF.number := LEFT.number, 
          SELF.class_number := LEFT.id, SELF.w := LEFT.value));
      Res := JOIN(Res0, SE, LEFT.number = RIGHT.number AND LEFT.class_number = RIGHT.id, 
        TRANSFORM(Logis_Model,
          SELF.Id := LEFT.Id,SELF.number := LEFT.number, 
          SELF.class_number := LEFT.class_number, SELF.w := LEFT.w, SELF.se := SQRT(RIGHT.value)));
          
      ToField(Res,o);
      EXPORT Mod := o;
      modelY_M := Mat.MU.From(BetaPair, mu_comp.Y);
      modelY_NF := Types.FromMatrix(modelY_M);
      EXPORT modelY := RebaseY.ToOld(modelY_NF, Y_Map);
    END;
    EXPORT LearnCS(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep) := Logis(Indep,PROJECT(Dep,Types.NumericField)).mod;
    EXPORT LearnC(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep) := LearnCConcat(Indep,Dep,LearnCS);
    EXPORT ClassifyS(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
      mod0 := Model(mod);
      Beta0 := PROJECT(mod0,TRANSFORM(Types.NumericField,SELF.Number := LEFT.Number+1,SELF.id := LEFT.class_number, SELF.value := LEFT.w;SELF:=LEFT;));
      mBeta := Types.ToMatrix(Beta0);
      mX_0 := Types.ToMatrix(Indep);
      mXloc := Mat.InsertColumn(mX_0, 1, 1.0); // Insert X1=1 column 

      AdjY := Mat.Mul(mXloc, Mat.Trans(mBeta)) ;
      // expy =  1./(1+exp(-adjy))
      sigmoid := Mat.Each.Reciprocal(Mat.Each.Add(Mat.Each.Exp(Mat.Scale(AdjY, -1)),1));
      // Now convert to classify return format
      Types.NumericField tr(sigmoid le) := TRANSFORM
        SELF.value := le.value;
        SELF.id := le.x;
        SELF.number := le.y;
      END;
      RETURN PROJECT(sigmoid,tr(LEFT));
    END;
      
  END; // Logistic_sparse Module
