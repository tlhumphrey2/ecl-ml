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
    - prows: an optional parameter used to set the number of rows in partition blocks (Should be used in conjuction with pcols)
    - pcols: an optional parameter used to set the number of cols in partition blocks (Should be used in conjuction with prows)
    - Maxrows: an optional parameter used to set maximum rows allowed per block when using AutoBVMap
    - Maxcols: an optional parameter used to set maximum cols allowed per block when using AutoBVMap

    The inputs to the Logis module are:
  a) A training dataset X of discretized independant variables
  b) A dataset of class results Y.

*/
  EXPORT Logistic(REAL8 Ridge=0.00001, REAL8 Epsilon=0.000000001, UNSIGNED2 MaxIter=200, 
    UNSIGNED4 prows=0, UNSIGNED4 pcols=0,UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE(Logistic_Model)

    Logis(DATASET(Types.NumericField) X, DATASET(Types.NumericField) Y) := MODULE
      SHARED mu_comp := ENUM ( Beta = 1,  Y = 2, BetaError = 3, BetaMaxError = 4, VC = 5 );
      SHARED RebaseY := Utils.RebaseNumericField(Y);
      SHARED Y_Map := RebaseY.Mapping(1);
      Y_0 := RebaseY.ToNew(Y_Map);
      SHARED mY := Types.ToMatrix(Y_0);
      mX_0 := Types.ToMatrix(X);
      SHARED mX := IF(NOT EXISTS(mX_0), 
                    Mat.Vec.ToCol(Mat.Vec.From(Mat.Has(mY).Stats.xmax, 1.0), 1), 
                    Mat.InsertColumn(mX_0, 1, 1.0)); // Insert X1=1 column
      mXstats := Mat.Has(mX).Stats;
      mX_n := mXstats.XMax;
      mX_m := mXstats.YMax;

      //Map for Matrix X. Map will be used to derive all other maps in Logis
      havemaxrow := maxrows > 0;
      havemaxcol := maxcols > 0;
      havemaxrowcol := havemaxrow and havemaxcol;

      derivemap := IF(havemaxrowcol, PBblas.AutoBVMap(mX_n, mX_m,prows,pcols,maxrows, maxcols),
        IF(havemaxrow, PBblas.AutoBVMap(mX_n, mX_m,prows,pcols,maxrows),
        IF(havemaxcol, PBblas.AutoBVMap(mX_n, mX_m,prows,pcols,,maxcols),
        PBblas.AutoBVMap(mX_n, mX_m,prows,pcols))));

      sizeRec := RECORD
        PBblas.Types.dimension_t m_rows;
        PBblas.Types.dimension_t m_cols;
        PBblas.Types.dimension_t f_b_rows;
        PBblas.Types.dimension_t f_b_cols;
      END;

      SHARED sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);


      mXmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
      //Create block matrix X
      mXdist := DMAT.Converted.FromElement(mX,mXmap);


      //Create block matrix Y
      mYmap := PBblas.Matrix_Map(sizeTable[1].m_rows, 1, sizeTable[1].f_b_rows, 1);
      mYdist := DMAT.Converted.FromElement(mY, mYmap);

      //New Matrix Generator
      Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows, REAL8 v) := TRANSFORM
      SELF.x := ((c-1) % NumRows) + 1;
      SELF.y := ((c-1) DIV NumRows) + 1;
      SELF.v := v;
      END;

      //Create block matrix W
      mW := DATASET(sizeTable[1].m_rows, gen(COUNTER, sizeTable[1].m_rows, 1.0),DISTRIBUTED);
      mWdist := DMAT.Converted.FromCells(mYmap, mW);

      //Create block matrix Ridge
      mRidge := DATASET(sizeTable[1].m_cols, gen(COUNTER, sizeTable[1].m_cols, ridge),DISTRIBUTED);
      RidgeVecMap := PBblas.Matrix_Map(sizeTable[1].m_cols, 1, sizeTable[1].f_b_cols, 1);
      Ridgemap := PBblas.Matrix_Map(sizeTable[1].m_cols, sizeTable[1].m_cols, sizeTable[1].f_b_cols, sizeTable[1].f_b_cols);
      mRidgeVec := DMAT.Converted.FromCells(RidgeVecMap, mRidge);
      mRidgedist := PBblas.Vector2Diag(RidgeVecMap, mRidgeVec, Ridgemap);

      //Create block matrix Beta
      mBeta0 := DATASET(sizeTable[1].m_cols, gen(COUNTER, sizeTable[1].m_cols, 0.0),DISTRIBUTED);
      mBeta0map := PBblas.Matrix_Map(sizeTable[1].m_cols, 1, sizeTable[1].f_b_cols, 1);
      mBeta00 := PBblas.MU.To(DMAT.Converted.FromCells(mBeta0map,mBeta0), mu_comp.Beta);

      //Create block matrix OldExpY
      OldExpY_0 := DATASET(sizeTable[1].m_rows, gen(COUNTER, sizeTable[1].m_rows, -1),DISTRIBUTED); // -ones(size(mY))
      OldExpY_00 := PBblas.MU.To(DMAT.Converted.FromCells(mYmap,OldExpY_0), mu_comp.Y);



      //Functions needed to calculate ExpY
      PBblas.Types.value_t e(PBblas.Types.value_t v, 
            PBblas.Types.dimension_t r, 
            PBblas.Types.dimension_t c) := exp(v);

      PBblas.Types.value_t AddOne(PBblas.Types.value_t v, 
            PBblas.Types.dimension_t r, 
            PBblas.Types.dimension_t c) := 1+v;

      PBblas.Types.value_t Reciprocal(PBblas.Types.value_t v, 
            PBblas.Types.dimension_t r, 
            PBblas.Types.dimension_t c) := 1/v;

      //Abs (Absolute Value) function
      PBblas.Types.value_t absv(PBblas.Types.value_t v, 
            PBblas.Types.dimension_t r, 
            PBblas.Types.dimension_t c) := abs(v);
            
      //Maps used in Step function
      weightsMap := PBblas.Matrix_Map(sizeTable[1].m_rows, sizeTable[1].m_rows, sizeTable[1].f_b_rows, sizeTable[1].f_b_rows);
      xWeightMap := PBblas.Matrix_Map(sizeTable[1].m_cols, sizeTable[1].m_rows, sizeTable[1].f_b_cols, sizeTable[1].f_b_rows);
      xtranswadjyMap := PBblas.Matrix_Map(sizeTable[1].m_cols, 1, sizeTable[1].f_b_cols, 1);

        
      Step(DATASET(PBblas.Types.MUElement) BetaPlusY, INTEGER coun) := FUNCTION

        OldExpY := PBblas.MU.From(BetaPlusY, mu_comp.Y);

        BetaDist := PBblas.MU.From(BetaPlusY, mu_comp.Beta);


        AdjY := PBblas.PB_dgemm(FALSE, FALSE, 1.0, mXmap, mXdist, mBeta0map, BetaDist,  mYmap);

        // The -adjy of expy =  1./(1+exp(-adjy))
        negAdjY := PBblas.PB_dscal(-1, AdjY);
        // The exp of expy =  1./(1+exp(-adjy))
        e2negAdjY := PBblas.Apply2Elements(mYmap, negAdjY, e);
        // The (1+exp(-adjy) of expy =  1./(1+exp(-adjy))
        OnePlusE2negAdjY := PBblas.Apply2Elements(mYmap, e2negAdjY, AddOne);

        // expy =  1./(1+exp(-adjy))
        ExpY := PBblas.Apply2Elements(mYmap, OnePlusE2negAdjY, Reciprocal); 

        // deriv := expy .* (1-expy)
        //prederiv := 
        Deriv := PBblas.HadamardProduct(mYmap, Expy, PBblas.Apply2Elements(mYmap,PBblas.PB_dscal(-1, Expy), AddOne));

        // Functions needed to calculate w_AdjY
        // The deriv .* adjy of wadjy := w .* (deriv .* adjy + (y-expy))
        derivXadjy := PBblas.HadamardProduct(mYmap, Deriv, AdjY);
        // The (y-expy) of wadjy := w .* (deriv .* adjy + (y-expy))
        yMINUSexpy := PBblas.PB_daxpy(1.0,mYdist,PBblas.PB_dscal(-1, Expy));
        // The (deriv .* adjy + (y-expy)) of wadjy := w .* (deriv .* adjy + (y-expy))
        forWadjy := PBblas.PB_daxpy(1, derivXadjy, yMINUSexpy);

        // wadjy := w .* (deriv .* adjy + (y-expy))
        w_Adjy := PBblas.HadamardProduct(mYmap, mWdist, forWadjy);

        // Functions needed to calculate Weights
        // The deriv .* w of weights := spdiags(deriv .* w, 0, n, n)
        derivXw := PBblas.HadamardProduct(mYmap,deriv, mWdist);

        // weights := spdiags(deriv .* w, 0, n, n)


        Weights := PBblas.Vector2Diag(weightsMap,derivXw,weightsMap);

        // Functions needed to calculate mBeta
        // x' * weights * x of mBeta := Inv(x' * weights * x + mRidge) * x' * wadjy

        xweight := PBblas.PB_dgemm(TRUE, FALSE, 1.0, mXmap, mXdist, weightsMap, weights, xWeightMap);
        xweightsx :=  PBblas.PB_dgemm(FALSE, FALSE, 1.0, xWeightMap, xweight, mXmap, mXdist, Ridgemap, mRidgedist, 1.0);

        // mBeta := Inv(x' * weights * x + mRidge) * x' * wadjy

        side := PBblas.PB_dgemm(TRUE, FALSE,1.0, mXmap, mXdist, mYmap, w_Adjy,xtranswadjyMap);

        LU_xwx  := PBblas.PB_dgetrf(Ridgemap, xweightsx);

        lc  := PBblas.PB_dtrsm(PBblas.Types.Side.Ax, PBblas.Types.Triangle.Lower, FALSE,
           PBblas.Types.Diagonal.UnitTri, 1.0, Ridgemap, LU_xwx, xtranswadjyMap, side);

        mBeta := PBblas.PB_dtrsm(PBblas.Types.Side.Ax, PBblas.Types.Triangle.Upper, FALSE,
             PBblas.Types.Diagonal.NotUnitTri, 1.0, Ridgemap, LU_xwx, xtranswadjyMap, lc);

        //Caculate error to be checked in loop evaluation
        err := SUM(DMAT.Converted.FromPart2Cell(PBblas.Apply2Elements(mBeta0map,PBblas.PB_daxpy(1.0, mBeta,PBblas.PB_dscal(-1, BetaDist)), absv)), v);

        errmap := PBblas.Matrix_Map(1, 1, 1, 1);

        BE := DATASET([{1,1,err}],Mat.Types.Element);
        BetaError := DMAT.Converted.FromElement(BE,errmap);

        BME := DATASET([{1,1,sizeTable[1].m_cols*Epsilon}],Mat.Types.Element);
        BetaMaxError := DMAT.Converted.FromElement(BME,errmap);         

        RETURN PBblas.MU.To(mBeta, mu_comp.Beta)
                  +PBblas.MU.To(ExpY, mu_comp.Y)
                  +PBblas.MU.To(BetaError,mu_comp.BetaError)
                  +PBblas.MU.To(BetaMaxError,mu_comp.BetaMaxError)
                  +PBblas.MU.To(xweightsx, mu_comp.VC);

      END;
      
      errmap := PBblas.Matrix_Map(1, 1, 1, 1);
      BE := DATASET([{1,1,sizeTable[1].m_cols*Epsilon+1}],Mat.Types.Element);
      BetaError00 := PBblas.MU.To(DMAT.Converted.FromElement(BE,errmap), mu_comp.BetaError);
      
      BME := DATASET([{1,1,sizeTable[1].m_cols*Epsilon}],Mat.Types.Element);
      BetaMaxError00 := PBblas.MU.To(DMAT.Converted.FromElement(BME,errmap), mu_comp.BetaMaxError); 
      SHARED BetaPair := LOOP(mBeta00+OldExpY_00+BetaError00+BetaMaxError00
            , (COUNTER<=MaxIter)
              AND (DMAT.Converted.FromPart2Elm(PBblas.MU.From(ROWS(LEFT),mu_comp.BetaError))[1].value > 
              DMAT.Converted.FromPart2Elm(PBblas.MU.From(ROWS(LEFT),mu_comp.BetaMaxError))[1].value)
            , Step(ROWS(LEFT),COUNTER)
            ); 

      SHARED mBeta00map := PBblas.Matrix_Map(sizeTable[1].m_cols, 1, sizeTable[1].f_b_cols, 1);  
      SHARED xwxmap := PBblas.Matrix_Map(sizeTable[1].m_cols, sizeTable[1].m_cols, sizeTable[1].f_b_cols, sizeTable[1].f_b_cols);
    
      EXPORT Beta := FUNCTION
        mubeta := DMAT.Converted.FromPart2DS(DMAT.Trans.Matrix(mBeta00map,PBblas.MU.From(BetaPair, mu_comp.Beta)));
        rebaseBeta := RebaseY.ToOldFromElemToPart(mubeta, Y_Map);
        RETURN rebaseBeta;
      END;
      
      EXPORT SE := FUNCTION
        mVC := DMat.Inv(xwxmap, PBblas.MU.From(BetaPair, mu_comp.VC));

        muSE := Types.FromMatrix(Mat.Trans(Mat.Vec.ToCol(Mat.Vec.FromDiag(
                    DMAT.Converted.FromPart2Elm(mVC)), 1)));
        rebaseSE := RebaseY.ToOldFromElemToPart(muSE, Y_Map);
        RETURN rebaseSE;
      END;

      Res := FUNCTION
        ret0 := PROJECT(Beta,TRANSFORM(Logis_Model,SELF.Id := COUNTER+Base,SELF.number := LEFT.number-1,
                              SELF.class_number := LEFT.id, SELF.w := LEFT.value, SELF.SE := 0.0));
        ret := JOIN(ret0, SE, LEFT.number+1 = RIGHT.number AND LEFT.class_number = RIGHT.id, 
                    TRANSFORM(Logis_Model,
                              SELF.Id := LEFT.Id,SELF.number := LEFT.number, 
                              SELF.class_number := LEFT.class_number, SELF.w := LEFT.w, SELF.se := SQRT(RIGHT.value)));
        RETURN ret;
      END;
      ToField(Res,o);

      EXPORT Mod := o;
      modelY_M := DMAT.Converted.FromPart2Elm(PBblas.MU.From(BetaPair, mu_comp.Y));
      modelY_NF := RebaseY.ToOld(Types.FromMatrix(modelY_M),Y_Map);
      EXPORT modelY := modelY_NF;
    END;//End Logis

    EXPORT LearnCS(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep) := Logis(Indep,PROJECT(Dep,Types.NumericField)).mod;
    EXPORT LearnC(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep) := LearnCConcat(Indep,Dep,LearnCS);
    EXPORT ClassifyS(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := FUNCTION

      mod0 := Model(mod);
      Beta_0 := PROJECT(mod0,TRANSFORM(Types.NumericField,SELF.Number := LEFT.Number+1,SELF.id := LEFT.class_number, SELF.value := LEFT.w;SELF:=LEFT;));
      RebaseBeta := Utils.RebaseNumericFieldID(Beta_0);
      Beta0_Map := RebaseBeta.MappingID(1);
      Beta0 := RebaseBeta.ToNew(Beta0_Map);

      mX_0 := Types.ToMatrix(Indep);
      mXloc := Mat.InsertColumn(mX_0, 1, 1.0); // Insert X1=1 column 

      mXlocstats := Mat.Has(mXloc).Stats;
      mXloc_n := mXlocstats.XMax;
      mXloc_m := mXlocstats.YMax;

      havemaxrow := maxrows > 0;
      havemaxcol := maxcols > 0;
      havemaxrowcol := havemaxrow and havemaxcol;

      //Map for Matrix X. Map will be used to derive all other maps in ClassifyC
      derivemap := IF(havemaxrowcol, PBblas.AutoBVMap(mXloc_n, mXloc_m,prows,pcols,maxrows, maxcols),
        IF(havemaxrow, PBblas.AutoBVMap(mXloc_n, mXloc_m,prows,pcols,maxrows),
        IF(havemaxcol, PBblas.AutoBVMap(mXloc_n, mXloc_m,prows,pcols,,maxcols),
        PBblas.AutoBVMap(mXloc_n, mXloc_m,prows,pcols))));


      sizeRec := RECORD
        PBblas.Types.dimension_t m_rows;
        PBblas.Types.dimension_t m_cols;
        PBblas.Types.dimension_t f_b_rows;
        PBblas.Types.dimension_t f_b_cols;
      END;

      sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);

      mXlocmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);

      mXlocdist := DMAT.Converted.FromElement(mXloc,mXlocmap);

      mBeta := Types.ToMatrix(Beta0);
      mBetastats := Mat.Has(mBeta).Stats;
      mBeta_n := mBetastats.XMax;

      mBetamap := PBblas.Matrix_Map(mBeta_n, sizeTable[1].m_cols, 1, sizeTable[1].f_b_cols);
      mBetadist := DMAT.Converted.FromElement(mBeta,mBetamap);

      AdjYmap := PBblas.Matrix_Map(mXlocmap.matrix_rows, mBeta_n, mXlocmap.part_rows(1), 1);
      AdjY := PBblas.PB_dgemm(FALSE, TRUE, 1.0, mXlocmap, mXlocdist, mBetamap, mBetaDist,AdjYmap);

      // expy =  1./(1+exp(-adjy))

      negAdjY := PBblas.PB_dscal(-1, AdjY);

      PBblas.Types.value_t e(PBblas.Types.value_t v, 
            PBblas.Types.dimension_t r, 
            PBblas.Types.dimension_t c) := exp(v);
            
      PBblas.Types.value_t AddOne(PBblas.Types.value_t v, 
            PBblas.Types.dimension_t r, 
            PBblas.Types.dimension_t c) := 1+v;
            
      PBblas.Types.value_t Reciprocal(PBblas.Types.value_t v, 
            PBblas.Types.dimension_t r, 
            PBblas.Types.dimension_t c) := 1/v;

      e2negAdjY := PBblas.Apply2Elements(AdjYmap, negAdjY, e);

      OnePlusE2negAdjY := PBblas.Apply2Elements(AdjYmap, e2negAdjY, AddOne);

      sig := PBblas.Apply2Elements(AdjYmap, OnePlusE2negAdjY, Reciprocal);

      //Rebase IDs so correct classifiers can be used
      sigtran := DMAT.Trans.Matrix(AdjYmap,sig);

      sigds :=DMAT.Converted.FromPart2DS(sigtran);

      sigconvds := RebaseBeta.ToOld(sigds, Beta0_Map);

      tranmap := PBblas.Matrix_Map(((mXloc_m-1)+mBeta_n), mXlocmap.matrix_rows, 1, mXlocmap.part_rows(1));

      preptranback := DMAT.Converted.FromNumericFieldDS(sigconvds, tranmap);

      sigtranback := DMAT.Trans.Matrix(tranmap, preptranback);

      sigmoid := DMAT.Converted.frompart2elm(sigtranback);

      // Now convert to classify return format
      Types.NumericField tr(sigmoid le) := TRANSFORM
        SELF.value := le.value;
        SELF.id := le.x;
        SELF.number := le.y;
      END;

      RETURN PROJECT(sigmoid,tr(LEFT));

    END;

  END; // Logistic Module 
