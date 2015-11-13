 IMPORT ML;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

/*
  See: http://en.wikipedia.org/wiki/Perceptron
  The inputs to the BuildPerceptron are:
  a) A dataset of discretized independant variables
  b) A dataset of class results (these must match in ID the discretized independant variables).
  c) Passes; number of passes over the data to make during the learning process
  d) Alpha is the learning rate - higher numbers may learn quicker - but may not converge
  Note the perceptron presently assumes the class values are ordinal eg 4>3>2>1>0

  Output: A table of weights for each independant variable for each class. 
  Those weights with number=class_number give the error rate on the last pass of the data
*/

  EXPORT Perceptron(UNSIGNED Passes,REAL8 Alpha = 0.1) := MODULE(DEFAULT)
    SHARED Thresh := 0.5; // The threshold to apply for the cut-off function

    EXPORT LearnD(DATASET(Types.DiscreteField) Indep,DATASET(Types.DiscreteField) Dep) := FUNCTION
      dd := Indep;
      cl := Dep;
      MaxFieldNumber := MAX(dd,number);
      FirstClassNo := MaxFieldNumber+1;
      clb := Utils.RebaseDiscrete(cl,FirstClassNo);
      LastClassNo := MAX(clb,number);
      all_fields := dd+clb;
  // Fields are ordered so that everything for a given input record is on one node
  // And so that records are encountered 'lowest first' and with the class variables coming later
      ready := SORT( DISTRIBUTE( all_fields, HASH(id) ), id, Number, LOCAL );
  // A weight record for our perceptron
      WR := RECORD
        REAL8 W := 0;
        Types.t_FieldNumber number; // The field this weight applies to - note field 0 will be the bias, class_number will be used for cumulative error
        Types.t_Discrete class_number;
      END;
      VR := RECORD
        Types.t_FieldNumber number;
        Types.t_Discrete    value;
      END;
  // This function exists to initialize the weights for the perceptron
      InitWeights := FUNCTION
        Classes := TABLE(clb,{number},number,FEW);
        WR again(Classes le,UNSIGNED C) := TRANSFORM
          SELF.number := IF( C > MaxFieldNumber, le.number, C ); // The > case sets up the cumulative error; rest are the field weights
          SELF.class_number := le.number;
        END;
        RETURN NORMALIZE(Classes,MaxFieldNumber+2,again(LEFT,COUNTER-1));
      END;

      AccumRec := RECORD
        DATASET(WR) Weights;
        DATASET(VR) ThisRecord;
        Types.t_RecordId Processed;
      END;
  // The learn step for a perceptrom
      Learn(DATASET(WR) le,DATASET(VR) ri,Types.t_FieldNumber fn,Types.t_Discrete va) := FUNCTION
        let := le(class_number=fn);
        letn := let(number<>fn);     // all of the regular weights
        lep := le(class_number<>fn); // Pass-thru
    // Compute the 'predicted' value for this iteration as Sum WiXi
        iv := RECORD
          REAL8 val;
        END;
    // Compute the score components for each class for this record
        iv scor(le l,ri r) := TRANSFORM
          SELF.val := l.w*IF(r.number<>0,r.value,1);
        END;
        sc := JOIN(letn,ri,LEFT.number=RIGHT.number,scor(LEFT,RIGHT),LEFT OUTER);
        res := IF( SUM(sc,val) > Thresh, 1, 0 );
        err := va-res;
        let_e := PROJECT(let(number=fn),TRANSFORM(WR,SELF.w := LEFT.w+ABS(err), SELF:=LEFT)); // Build up the accumulative error
        delta := alpha*err; // The amount of 'learning' to do this step
    // Apply delta to regular weights
        WR add(WR le,VR ri) := TRANSFORM
          SELF.w := le.w+delta*IF(ri.number=0,1,ri.value); // Bias will not have matching RHS - so assume 1
          SELF := le;
        END;
        J := JOIN(letn,ri,LEFT.number=right.number,add(LEFT,RIGHT),LEFT OUTER);
        RETURN let_e+J+lep;
      END;
  // Zero out the error values
      WR Clean(DATASET(WR) w) := FUNCTION
        RETURN w(number<>class_number)+PROJECT(w(number=class_number),TRANSFORM(WR,SELF.w := 0, SELF := LEFT));
      END;
  // This function does one pass of the data learning into the weights
      WR Pass(DATASET(WR) we) := FUNCTION
    // This takes a record one by one and processes it
    // That may mean simply appending it to 'ThisRecord' - or it might mean performing a learning step
        AccumRec TakeRecord(ready le,AccumRec ri) := TRANSFORM
          BOOLEAN lrn := le.number >= FirstClassNo;
          BOOLEAN init := ~EXISTS(ri.Weights);
          SELF.Weights := MAP ( init => Clean(we), 
                                ~lrn => ri.Weights,
                                Learn(ri.Weights,ri.ThisRecord,le.number,le.value) );
    // This is either an independant variable - in which case we append it
    // Or it is the last dependant variable - in which case we can throw the record away
    // Or it is one of the dependant variables - so keep the record for now
          SELF.ThisRecord := MAP ( ~lrn => ri.ThisRecord+ROW({le.number,le.value},VR),
                                  le.number = LastClassNo => DATASET([],VR),
                                  ri.ThisRecord);
          SELF.Processed := ri.Processed + IF( le.number = LastClassNo, 1, 0 );
        END;
      // Effectively merges two perceptrons (generally 'learnt' on different nodes)
      // For the errors - simply add them
      // For the weights themselves perform a weighted mean (weighting by the number of records used to train)
        Blend(DATASET(WR) l,UNSIGNED lscale, DATASET(WR) r,UNSIGNED rscale) := FUNCTION
          lscaled := PROJECT(l(number<>class_number),TRANSFORM(WR,SELF.w := LEFT.w*lscale, SELF := LEFT));
          rscaled := PROJECT(r(number<>class_number),TRANSFORM(WR,SELF.w := LEFT.w*rscale, SELF := LEFT));
          unscaled := (l+r)(number=class_number);
          t := TABLE(lscaled+rscaled+unscaled,{number,class_number,w1 := SUM(GROUP,w)},number,class_number,FEW);
          RETURN PROJECT(t,TRANSFORM(WR,SELF.w := IF(LEFT.number=LEFT.class_number,LEFT.w1,LEFT.w1/(lscale+rscale)),SELF := LEFT));
        END;    
        AccumRec MergeP(AccumRec ri1,AccumRec ri2) := TRANSFORM
          SELF.ThisRecord := []; // Merging only valid across perceptrons learnt on complete records
          SELF.Processed := ri1.Processed+ri2.Processed;
          SELF.Weights := Blend(ri1.Weights,ri1.Processed,ri2.Weights,ri2.Processed);
        END;

        A := AGGREGATE(ready,AccumRec,TakeRecord(LEFT,RIGHT),MergeP(RIGHT1,RIGHT2))[1];
    // Now return the weights (and turn the error number into a ratio)
        RETURN A.Weights(class_number<>number)+PROJECT(A.Weights(class_number=number),TRANSFORM(WR,SELF.w := LEFT.w / A.Processed,SELF := LEFT));
      END;
      L := LOOP(InitWeights,Passes,PASS(ROWS(LEFT)));
      L1 := PROJECT(L,TRANSFORM(l_model,SELF.id := COUNTER+Base,SELF := LEFT));
      ML.ToField(L1,o);
      RETURN o;
    END;
    EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
      ML.FromField(mod,l_model,o);
      RETURN o;
    END;
    EXPORT ClassifyD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
      mo := Model(mod);
      Ind := DISTRIBUTE(Indep,HASH(id));
      l_result note(Ind le,mo ri) := TRANSFORM
        SELF.conf := le.value*ri.w;
        SELF.number := ri.class_number;
        SELF.value := 0;
        SELF.id := le.id;
      END;
      // Compute the score for each component of the linear equation
      j := JOIN(Ind,mo,LEFT.number=RIGHT.number,note(LEFT,RIGHT),MANY LOOKUP); // MUST be lookup! Or distribution goes
      l_result ac(l_result le, l_result ri) := TRANSFORM
        SELF.conf := le.conf+ri.conf;
        SELF := le;
      END;
      // Rollup so there is one score for every id for every 'number' (original class_number)
      t := ROLLUP(SORT(j,id,number,LOCAL),LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,ac(LEFT,RIGHT),LOCAL);
      // Now we have to add on the 'constant' offset
      l_result add_c(l_result le,mo ri) := TRANSFORM
        SELF.conf := le.conf+ri.w;
        SELF.value := IF(SELF.Conf>Thresh,1,0);
        SELF := le;
      END;
      t1 := JOIN(t,mo(number=0),LEFT.number=RIGHT.class_number,add_c(LEFT,RIGHT),LEFT OUTER);
      t2 := PROJECT(t1,TRANSFORM(l_Result,SELF.conf := ABS(LEFT.Conf-Thresh), SELF := LEFT));
      RETURN t2;
    END;
  END;
