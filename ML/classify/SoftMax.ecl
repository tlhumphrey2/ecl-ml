 IMPORT ML;
IMPORT * FROM ML;
IMPORT * FROM ML.Mat;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

/*
// Implementation of SoftMax classifier using PBblas Library
//SoftMax classifier generalizes logistic regression classifier for cases when we have more than two target classes
//The implemenataion is based on Stanford Deep Learning tutorial availabe at http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
//this implementation is based on using PBblas library
//parameters:
//LAMBDA : wight decay parameter in calculating SoftMax costfunction
//ALPHA : learning rate for updating softmax parameters
//IntTHETA: Initialized parameters that is a matrix of size (number of classes) * (number of features)
*/
EXPORT SoftMax(DATASET (MAT.Types.Element) IntTHETA, REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100,
  UNSIGNED4 prows=0, UNSIGNED4 pcols=0,UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE(DEFAULT)
  Soft(DATASET(Types.NumericField) X,DATASET(Types.NumericField) Y) := MODULE
  //Convert the input data to matrix
  //the reason matrix transform is done after ocnverting the input data to the matrix is that
  //in this implementation it is assumed that the  input matrix shows the samples in column-wise format
  //in other words each sample is shown in one column. that's why after converting the input to matrix we apply
  //matrix tranform to refletc samples in column-wise format
  dt := Types.ToMatrix (X);
  SHARED dTmp := Mat.InsertColumn(dt,1,1.0); // add the intercept column
  SHARED d := Mat.Trans(dTmp); //in the entire of the calculations we work with the d matrix that each sample in presented in one column
  SHARED groundTruth:= Utils.ToGroundTruth (Y);//Instead of working with label matrix we work with groundTruth matrix
  //groundTruth is a Numclass*NumSamples matrix. groundTruth(i,j)=1 if label of the jth sample is i, otherwise groundTruth(i,j)=0
  SHARED NumClass := Mat.Has(groundTruth).Stats.XMax;
  SHARED sizeRec := RECORD
    PBblas.Types.dimension_t m_rows;
    PBblas.Types.dimension_t m_cols;
    PBblas.Types.dimension_t f_b_rows;
    PBblas.Types.dimension_t f_b_cols;
  END;
   //Map for Matrix d.
    SHARED havemaxrow := maxrows > 0;
    SHARED havemaxcol := maxcols > 0;
    SHARED havemaxrowcol := havemaxrow and havemaxcol;
    SHARED dstats := Mat.Has(d).Stats;
    SHARED d_n := dstats.XMax;
    SHARED d_m := dstats.YMax;
    derivemap := IF(havemaxrowcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows, maxcols),
                   IF(havemaxrow, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows),
                      IF(havemaxcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,,maxcols),
                      PBblas.AutoBVMap(d_n, d_m,prows,pcols))));
    SHARED sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);
    Ones_VecMap := PBblas.Matrix_Map(1, NumClass, 1, NumClass);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.y := ((c-1) % NumRows) + 1;
      SELF.x := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(NumClass, gen(COUNTER, NumClass));
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    //Create block matrix d
    dmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
    ddist := DMAT.Converted.FromElement(d,dmap);
    //Create block matrix groundTruth
    groundTruthmap := PBblas.Matrix_Map(NumClass, sizeTable[1].m_cols, NumClass, sizeTable[1].f_b_cols);
    groundTruthdist := DMAT.Converted.FromElement(groundTruth, groundTruthmap);
    // creat block matrix for IntTHETA
    IntTHETAmap := PBblas.Matrix_Map(NumClass, sizeTable[1].m_rows, NumClass, sizeTable[1].f_b_rows);
    IntTHETAdist := DMAT.Converted.FromElement(IntTHETA, IntTHETAmap);
    //Maps used in step fucntion
    col_col_map := PBblas.Matrix_Map(sizeTable[1].m_cols, sizeTable[1].m_cols, sizeTable[1].f_b_cols, sizeTable[1].f_b_cols);
    THETAmap := PBblas.Matrix_Map(NumClass, sizeTable[1].m_rows, NumClass, sizeTable[1].f_b_rows);
    txmap := groundTruthmap;
    SumColMap := PBblas.Matrix_Map(1, sizeTable[1].m_cols, 1, sizeTable[1].f_b_cols);
    //functions used in step fucntion
    PBblas.Types.value_t e(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := exp(v);
    PBblas.Types.value_t reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := 1/v;
    m := d_m; //number of samples
    m_1 := -1 * (1/m);
    Step(DATASET(PBblas.Types.Layout_Part ) THETA) := FUNCTION
      // tx=(theta*d);
      tx := PBblas.PB_dgemm(FALSE, FALSE, 1.0, THETAmap, THETA, dmap, ddist, txmap);
      // tx_M = bsxfun(@minus, tx, max(tx, [], 1));
      tx_mat := DMat.Converted.FromPart2Elm(tx);
      MaxCol_tx_mat := Mat.Has(tx_mat).MaxCol;
      MaxCol_tx := DMAT.Converted.FromElement(MaxCol_tx_mat, SumColMap);
      tx_M := PBblas.PB_dgemm(TRUE, FALSE, -1.0, Ones_VecMap, Ones_Vecdist, SumColMap, MaxCol_tx, txmap, tx, 1.0);
      // Mat.Types.Element DoMinus(tx_mat le,MaxCol_tx_mat ri) := TRANSFORM
        // SELF.x := le.x;
        // SELF.y := le.y;
        // SELF.value := le.value - ri.value;
      // END;
      // tx_mat_M :=  JOIN(tx_mat, MaxCol_tx_mat, LEFT.y=RIGHT.y, DoMinus(LEFT,RIGHT),LOOKUP);
      // tx_M := DMAT.Converted.FromElement(tx_mat_M, txmap);
      //exp_tx_M=exp(tx_M);
      exp_tx_M := PBblas.Apply2Elements(txmap, tx_M, e);
      //Prob = bsxfun(@rdivide, exp_tx_M, sum(exp_tx_M));
      SumCol_exp_tx_M := PBblas.PB_dgemm(FALSE, FALSE, 1.0, Ones_VecMap, Ones_Vecdist, txmap, exp_tx_M, SumColMap);
      SumCol_exp_tx_M_rcip := PBblas.Apply2Elements(SumColMap, SumCol_exp_tx_M, Reci);
      SumCol_exp_tx_M_rcip_diag := PBblas.Vector2Diag(SumColMap, SumCol_exp_tx_M_rcip, col_col_map);
      Prob := PBblas.PB_dgemm(FALSE, FALSE, 1.0, txmap, exp_tx_M, col_col_map, SumCol_exp_tx_M_rcip_diag, txmap);
      second_term := PBblas.PB_dscal((1-ALPHA*LAMBDA), THETA);
      groundTruth_Prob := PBblas.PB_daxpy(1.0,groundTruthdist,PBblas.PB_dscal(-1, Prob));
      groundTruth_Prob_x := PBblas.PB_dgemm(FALSE, True, 1.0, txmap, groundTruth_Prob, dmap, ddist, THETAmap);
      // first_term := PBblas.PB_dscal((-1*ALPHA*m_1), groundTruth_Prob_x);
      // UpdatedTHETA := PBblas.PB_daxpy(1.0, first_term, second_term);
      UpdatedTHETA := PBblas.PB_daxpy((-1*ALPHA*m_1), groundTruth_Prob_x, second_term);
      RETURN UpdatedTHETA;
    END; // END step
    param := LOOP(IntTHETAdist, COUNTER <= MaxIter, Step(ROWS(LEFT)));
    //param := LOOP(IntTHETAdist, MaxIter, Step(ROWS(LEFT))); // does not work
    EXPORT Mod := ML.DMat.Converted.FromPart2DS (param);
  END; //END Soft
  EXPORT LearnC(DATASET(Types.NumericField) Indep, DATASET(Types.DiscreteField) Dep) := Soft(Indep,PROJECT(Dep,Types.NumericField)).mod;
  EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
    o:= Types.ToMatrix (Mod);
    RETURN o;
  END; // END Model
  EXPORT ClassProbDistribC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) :=FUNCTION
    // take the same steps take in step function to calculate prob
    X := Indep;
    dt := Types.ToMatrix (X);
    dTmp := Mat.InsertColumn(dt,1,1.0);
    d := Mat.Trans(dTmp);
    sizeRec := RECORD
      PBblas.Types.dimension_t m_rows;
      PBblas.Types.dimension_t m_cols;
      PBblas.Types.dimension_t f_b_rows;
      PBblas.Types.dimension_t f_b_cols;
    END;
   //Map for Matrix d.
    havemaxrow := maxrows > 0;
    havemaxcol := maxcols > 0;
    havemaxrowcol := havemaxrow and havemaxcol;
    dstats := Mat.Has(d).Stats;
    d_n := dstats.XMax;
    d_m := dstats.YMax;
    derivemap := IF(havemaxrowcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows, maxcols),
                   IF(havemaxrow, PBblas.AutoBVMap(d_n, d_m,prows,pcols,maxrows),
                      IF(havemaxcol, PBblas.AutoBVMap(d_n, d_m,prows,pcols,,maxcols),
                      PBblas.AutoBVMap(d_n, d_m,prows,pcols))));
    sizeTable := DATASET([{derivemap.matrix_rows,derivemap.matrix_cols,derivemap.part_rows(1),derivemap.part_cols(1)}], sizeRec);
    dmap := PBblas.Matrix_Map(sizeTable[1].m_rows,sizeTable[1].m_cols,sizeTable[1].f_b_rows,sizeTable[1].f_b_cols);
    //Create block matrix d
    ddist := DMAT.Converted.FromElement(d,dmap);
    param := Model (mod);
    NumClass := Mat.Has(param).Stats.XMax;
    Ones_VecMap := PBblas.Matrix_Map(1, NumClass, 1, NumClass);
    //New Vector Generator
    Layout_Cell gen(UNSIGNED4 c, UNSIGNED4 NumRows) := TRANSFORM
      SELF.y := ((c-1) % NumRows) + 1;
      SELF.x := ((c-1) DIV NumRows) + 1;
      SELF.v := 1;
    END;
    //Create Ones Vector for the calculations in the step fucntion
    Ones_Vec := DATASET(NumClass, gen(COUNTER, NumClass),DISTRIBUTED);
    Ones_Vecdist := DMAT.Converted.FromCells(Ones_VecMap, Ones_Vec);
    THETAmap := PBblas.Matrix_Map(NumClass, sizeTable[1].m_rows, NumClass, sizeTable[1].f_b_rows);
    THETA := DMAT.Converted.FromElement(param, THETAmap);
    txmap := PBblas.Matrix_Map(NumClass, sizeTable[1].m_cols, NumClass, sizeTable[1].f_b_cols);
    SumColMap := PBblas.Matrix_Map(1, sizeTable[1].m_cols, 1, sizeTable[1].f_b_cols);
    col_col_map := PBblas.Matrix_Map(sizeTable[1].m_cols, sizeTable[1].m_cols, sizeTable[1].f_b_cols, sizeTable[1].f_b_cols);
    PBblas.Types.value_t reci(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := 1/v;
    tx := PBblas.PB_dgemm(FALSE, FALSE, 1.0, THETAmap, THETA, dmap, ddist, txmap);
    // tx_M = bsxfun(@minus, tx, max(tx, [], 1));
    tx_mat := DMat.Converted.FromPart2Elm(tx);
    MaxCol_tx_mat := Mat.Has(tx_mat).MaxCol;
    MaxCol_tx := DMAT.Converted.FromElement(MaxCol_tx_mat, SumColMap);
    tx_M := PBblas.PB_dgemm(TRUE, FALSE, -1.0, Ones_VecMap, Ones_Vecdist, SumColMap, MaxCol_tx, txmap, tx, 1.0);
    //exp_tx_M=exp(tx_M);
    PBblas.Types.value_t e(PBblas.Types.value_t v,PBblas.Types.dimension_t r,PBblas.Types.dimension_t c) := exp(v);
    //Prob = bsxfun(@rdivide, exp_tx_M, sum(exp_tx_M));
    exp_tx_M := PBblas.Apply2Elements(txmap, tx_M, e);
    SumCol_exp_tx_M := PBblas.PB_dgemm(FALSE, FALSE, 1.0, Ones_VecMap, Ones_Vecdist, txmap, exp_tx_M, SumColMap);
    SumCol_exp_tx_M_rcip := PBblas.Apply2Elements(SumColMap, SumCol_exp_tx_M, Reci);
    SumCol_exp_tx_M_rcip_diag := PBblas.Vector2Diag(SumColMap, SumCol_exp_tx_M_rcip, col_col_map);
    Prob := PBblas.PB_dgemm(FALSE, FALSE, 1.0, txmap, exp_tx_M, col_col_map, SumCol_exp_tx_M_rcip_diag, txmap);
    Prob_mat := DMAT.Converted.FromPart2Elm (Prob);
    Types.l_result tr(Mat.Types.Element le) := TRANSFORM
      SELF.value := le.x;
      SELF.id := le.y;
      SELF.number := 1; //number of class
      SELF.conf := le.value;
    END;
    RETURN PROJECT (Prob_mat, tr(LEFT));
  END; // END ClassProbDistribC Function
  EXPORT ClassifyC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
    Dist := ClassProbDistribC(Indep, mod);
    numrow := MAX (Dist,Dist.value);
    S:= SORT(Dist,id,conf);
    SeqRec := RECORD
    l_result;
    INTEGER8 Sequence := 0;
    END;
    //add seq field to S
    SeqRec AddS (S l, INTEGER c) := TRANSFORM
    SELF.Sequence := c%numrow;
    SELF := l;
    END;
    Sseq := PROJECT(S, AddS(LEFT,COUNTER));
    classified := Sseq (Sseq.Sequence=0);
    RETURN PROJECT(classified,l_result);
  END; // END ClassifyC Function
END; //END SoftMax
