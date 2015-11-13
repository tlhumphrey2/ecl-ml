 IMPORT ML;
 IMPORT * FROM ML;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

  EXPORT Logistic_Model := MODULE(DEFAULT), VIRTUAL
    SHARED Logis_Model := RECORD(l_model)
      REAL8 SE;
    END;
    
    SHARED ZStat_Model := RECORD(Logis_Model)
      REAL8 z;
      REAL8 pVal;
    END;
    
    SHARED ConfInt_Model := RECORD(l_model)
      REAL8 lowerCI;
      REAL8 UpperCI;
    END;
    
    SHARED DevianceRec := RECORD
      Types.t_RecordID id;
      Types.t_FieldNumber classifier;  // The classifier in question (value of 'number' on outcome data)
      Types.t_Discrete  c_actual;      // The value of c provided
      Types.t_FieldReal  c_modeled;    // The value produced by the classifier
      Types.t_FieldReal LL;         // Score allocated by classifier
      BOOLEAN isGreater;
    END;
    
    SHARED ResidDevRec := RECORD
      Types.t_FieldNumber classifier;
      REAL8 Deviance;
      UNSIGNED4 DF;
    END;
    
    EXPORT LearnCS(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep) := DATASET([], Types.NumericField);
    EXPORT LearnC(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep) := LearnCConcat(Indep,Dep,LearnCS);
    EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
      FromField(mod,Logis_Model,o);
      RETURN o;
    END;  
    
    SHARED norm_dist := ML.Distribution.Normal(0, 1);
    ZStat_Model zStat_Transform(Logis_Model mod) := TRANSFORM
      z := mod.w/mod.SE;
      SELF.z := z;
      SELF.pVal := 2 * (1 - norm_dist.Cumulative(ABS(z)));
      SELF := mod;
    END;
    EXPORT ZStat(DATASET(Types.NumericField) mod) := PROJECT(Model(mod), zStat_Transform(LEFT));
    
    confInt_Model confint_transform(Logis_Model b, REAL Margin) := TRANSFORM
      SELF.UpperCI := b.w + Margin * b.se;
      SELF.LowerCI := b.w - Margin * b.se;
      SELF := b;
    END;    
    EXPORT ConfInt(Types.t_fieldReal level, DATASET(Types.NumericField) mod) := FUNCTION
      newlevel := 100 - (100 - level)/2;
      Margin := norm_dist.NTile(newlevel);
      RETURN PROJECT(Model(mod),confint_transform(LEFT,Margin));
    END;
    
    EXPORT ClassifyS(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := DATASET([], Types.NumericField);   
    l_result tr(Types.NumericField le) := TRANSFORM
        SELF.value := IF ( le.value > 0.5,1,0);
        SELF.conf := ABS(le.value-0.5);
        SELF := le;
      END;
    EXPORT ClassifyC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := PROJECT(ClassifyS(Indep, mod),tr(LEFT));
    
    DevianceRec  dev_t(Types.NumericField le, Types.DiscreteField ri) := TRANSFORM
      SELF.c_actual := ri.value;
      SELF.c_modeled := le.value;
      SELF.LL := -2 * (ri.value * LN(le.value) + (1 - ri.Value) * LN(1-le.Value));
      SELF.classifier := ri.number;
      SELF.id := ri.id;
      SELF.isGreater := IF(ri.value >= le.value, TRUE, FALSE);
    END;
    
    DevianceRec  dev_t2(REAL mu, Types.DiscreteField ri) := TRANSFORM
      SELF.c_actual := ri.value;
      SELF.c_modeled := mu;
      SELF.LL := -2 * (ri.value * LN(mu) + (1 - ri.Value) * LN(1-mu));
      SELF.classifier := ri.number;
      SELF.id := ri.id;
      SELF.isGreater := IF(ri.value >= mu, TRUE, FALSE);
    END;
    
    EXPORT DevianceC(DATASET(Types.NumericField) Indep,DATASET(Types.DiscreteField) Dep, DATASET(Types.NumericField) mod) := MODULE
      SHARED Dev := JOIN(ClassifyS(Indep, mod), Dep,LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number,dev_t(LEFT,RIGHT));
      NullMu := TABLE(Dep, {number, Mu := AVE(GROUP, value)}, number, FEW, UNSORTED);
      SHARED NDev := JOIN(Dep, NullMu, LEFT.number = RIGHT.number, dev_t2(RIGHT.Mu, LEFT),LOOKUP); 
      EXPORT DevRes := PROJECT(Dev, TRANSFORM(DevianceRec, SELF.LL := SQRT(LEFT.LL) * IF(LEFT.isGreater, +1, -1); SELF := LEFT));
      EXPORT DevNull := PROJECT(NDev, TRANSFORM(DevianceRec, SELF.LL := SQRT(LEFT.LL) * IF(LEFT.isGreater, +1, -1); SELF := LEFT));
      SHARED p := MAX(Model(mod), number) + 1;
      EXPORT ResidDev := PROJECT(TABLE(Dev, {classifier, Deviance := SUM(GROUP, LL), DF := COUNT(GROUP) - p}, classifier, FEW, UNSORTED), ResidDevRec);
      EXPORT NullDev := PROJECT(TABLE(NDev, {classifier, Deviance := SUM(GROUP, LL), DF := COUNT(GROUP) - 1}, classifier, FEW, UNSORTED), ResidDevRec);
      EXPORT AIC := PROJECT(ResidDev, TRANSFORM({Types.t_FieldNumber classifier, REAL8 AIC}, 
                                                SELF.AIC := LEFT.Deviance + 2 * p; SELF := LEFT));
    END;
    
    EXPORT AOD(DATASET(ResidDevRec) R1, DATASET(ResidDevRec) R2) := FUNCTION
      AODRec := RECORD
        Types.t_FieldNumber classifier;
        UNSIGNED4 ResidDF;
        REAL8 ResidDeviance;
        UNSIGNED4 DF := 0.0;
        REAL8 Deviance := 0.0;
        REAL8 pValue := 0.0;
      END;
      c1 := PROJECT(R1, TRANSFORM(AODRec, SELF.ResidDF := LEFT.DF; SELF.ResidDeviance := LEFT.Deviance; SELF.classifier := LEFT.classifier));
      c2 := JOIN(R1, R2, LEFT.classifier = RIGHT.classifier, 
                      TRANSFORM(AODRec,     df := LEFT.DF - RIGHT.DF;
                                            dev := LEFT.Deviance - RIGHT.Deviance;
                                            dist := ML.Distribution.ChiSquare(df);
                                            SELF.ResidDF := RIGHT.DF; 
                                            SELF.ResidDeviance := RIGHT.Deviance; 
                                            SELF.classifier := RIGHT.classifier;
                                            SELF.DF := df; 
                                            SELF.Deviance := dev;
                                            SELF.pValue := ( 1 - dist.Cumulative(ABS(dev)))), LOOKUP);
      RETURN c1+c2;
    END;
      
  
    EXPORT DevianceD(DATASET(Types.DiscreteField) Indep, DATASET(Types.DiscreteField) Dep, DATASET(Types.NumericField) mod) :=
      DevianceC(PROJECT(Indep, Types.NumericField), Dep, mod);
  END;
