 IMPORT ML;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

/*
  The purpose of this module is to provide a default interface to provide access to any of the 
  classifiers
*/
  EXPORT Default := MODULE,VIRTUAL
    EXPORT Base := 1000; // ID Base - all ids should be higher than this
    // Premise - two models can be combined by concatenating (in terms of ID number) the under-base and over-base parts
    SHARED CombineModels(DATASET(Types.NumericField) sofar,DATASET(Types.NumericField) new) := FUNCTION
      UBaseHigh := MAX(sofar(id<Base),id);
      High := IF(EXISTS(sofar),MAX(sofar,id),Base);
      UB := PROJECT(new(id<Base),TRANSFORM(Types.NumericField,SELF.id := LEFT.id+UBaseHigh,SELF := LEFT));
      UO := PROJECT(new(id>=Base),TRANSFORM(Types.NumericField,SELF.id := LEFT.id+High-Base,SELF := LEFT));
      RETURN sofar+UB+UO;
    END;
    // Learn from continuous data
    EXPORT LearnC(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep) := DATASET([],Types.NumericField); // All classifiers serialized to numeric field format
    // Learn from discrete data - worst case - convert to continuous
    EXPORT LearnD(DATASET(Types.DiscreteField) Indep,DATASET(Types.DiscreteField) Dep) := LearnC(PROJECT(Indep,Types.NumericField),Dep);
    // Learn from continuous data - using a prebuilt model
    EXPORT ClassifyC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := DATASET([],l_result);
    // Classify discrete data - using a prebuilt model
    EXPORT ClassifyD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod) := ClassifyC(PROJECT(Indep,Types.NumericField),mod);
    EXPORT TestD(DATASET(Types.DiscreteField) Indep,DATASET(Types.DiscreteField) Dep) := FUNCTION
      a := LearnD(Indep,Dep);
      res := ClassifyD(Indep,a);
      RETURN Compare(Dep,res);
    END;
    EXPORT TestC(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep) := FUNCTION
      a := LearnC(Indep,Dep);
      res := ClassifyC(Indep,a);
      RETURN Compare(Dep,res);
    END;
    EXPORT LearnDConcat(DATASET(Types.DiscreteField) Indep,DATASET(Types.DiscreteField) Dep, LearnD fnc) := FUNCTION
      // Call fnc once for each dependency; concatenate the results
      // First get all the dependant numbers
      dn := DEDUP(Dep,number,ALL);
      Types.NumericField loopBody(DATASET(Types.NumericField) sf,UNSIGNED c) := FUNCTION
        RETURN CombineModels(sf,fnc(Indep,Dep(number=dn[c].number)));
      END;
      RETURN LOOP(DATASET([],Types.NumericField),COUNT(dn),loopBody(ROWS(LEFT),COUNTER));
    END;
    EXPORT LearnCConcat(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep, LearnC fnc) := FUNCTION
      // Call fnc once for each dependency; concatenate the results
      // First get all the dependant numbers
      dn := DEDUP(Dep,number,ALL);
      Types.NumericField loopBody(DATASET(Types.NumericField) sf,UNSIGNED c) := FUNCTION
        RETURN CombineModels(sf,fnc(Indep,Dep(number=dn[c].number)));
      END;
      RETURN LOOP(DATASET([],Types.NumericField),COUNT(dn),loopBody(ROWS(LEFT),COUNTER));
    END;
  END;
