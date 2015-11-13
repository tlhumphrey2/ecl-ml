 IMPORT ML;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

/*
  Area Under the ROC curve
  // The function calculate the Area Under the ROC curve based on:
  // - classProbDistclass : probability distribution for each instance
  // - positiveClass      : the class of interest
  // - Dep                : instance's class value
  // The function returns all points of the ROC curve for graphic purposes:
  // label: threshold, point: (threshold's false negative rate, threshold's true positive rate).
  // The area under the ROC curve is returned in the AUC field of the last record.
  // Note: threshold = 100 means classifying all instances as negative, it is not necessarily part of the curve
*/
  EXPORT AUC_ROC(DATASET(l_result) classProbDist, Types.t_Discrete positiveClass, DATASET(Types.DiscreteField) Dep) := FUNCTION
    SHARED cntREC:= RECORD
      Types.t_FieldNumber classifier;  // The classifier in question (value of 'number' on outcome data)
      Types.t_Discrete  c_actual;      // The value of c provided
      Types.t_FieldReal score :=-1;
      Types.t_count     tp_cnt:=0;
      Types.t_count     fn_cnt:=0;
      Types.t_count     fp_cnt:=0;
      Types.t_count     tn_cnt:=0;
    END;
    SHARED compREC:= RECORD(cntREC)
      Types.t_Discrete  c_modeled;
    END;
    classOfInterest := classProbDist(value = positiveClass);
    compared:= JOIN(classOfInterest, Dep, LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,
                            TRANSFORM(compREC, SELF.classifier:= LEFT.number, SELF.c_actual:=RIGHT.value,
                            SELF.c_modeled:=LEFT.value, SELF.score:=LEFT.conf), HASH);
    sortComp:= SORT(compared, score);
    coi_acc:= TABLE(sortComp, {classifier, score, cntPos:= COUNT(GROUP, c_actual = c_modeled),
                                  cntNeg:= COUNT(GROUP, c_actual<>c_modeled)}, classifier, score, LOCAL);
    coi_tot:= TABLE(coi_acc, {classifier, totPos:= SUM(GROUP, cntPos), totNeg:= SUM(GROUP, cntNeg)}, classifier, FEW);
    totPos:=EVALUATE(coi_tot[1], totPos);
    totNeg:=EVALUATE(coi_tot[1], totNeg);
    // Count and accumulate number of TP, FP, TN and FN instances for each threshold (score)
    acc_sorted:= PROJECT(coi_acc, TRANSFORM(cntREC, SELF.c_actual:= positiveClass, SELF.fn_cnt:= LEFT.cntPos,
                                  SELF.tn_cnt:= LEFT.cntNeg, SELF:= LEFT), LOCAL);
    cntREC accNegPos(cntREC l, cntREC r) := TRANSFORM
      deltaPos:= l.fn_cnt + r.fn_cnt;
      deltaNeg:= l.tn_cnt + r.tn_cnt;
      SELF.score:= r.score;
      SELF.tp_cnt:=  totPos - deltaPos;
      SELF.fn_cnt:=  deltaPos;
      SELF.fp_cnt:=  totNeg - deltaNeg;
      SELF.tn_cnt:= deltaNeg;
      SELF:= r;
    END;
    cntNegPos:= ITERATE(acc_sorted, accNegPos(LEFT, RIGHT));
    accnew := DATASET([{1,positiveClass,-1,totPos,0,totNeg,0}], cntREC) + cntNegPos;
    curvePoint:= RECORD
      Types.t_Count       id;
      Types.t_FieldNumber classifier;
      Types.t_FieldReal   thresho;
      Types.t_FieldReal   fpr;
      Types.t_FieldReal   tpr;
      Types.t_FieldReal   deltaPos:=0;
      Types.t_FieldReal   deltaNeg:=0;
      Types.t_FieldReal   cumNeg:=0;
      Types.t_FieldReal   AUC:=0;
    END;
    // Transform all into ROC curve points
    rocPoints:= PROJECT(accnew, TRANSFORM(curvePoint, SELF.id:=COUNTER, SELF.thresho:=LEFT.score,
                                SELF.fpr:= LEFT.fp_cnt/totNeg, SELF.tpr:= LEFT.tp_cnt/totPos, SELF.AUC:=IF(totNeg=0,1,0) ,SELF:=LEFT));
    // Calculate the area under the curve (cumulative iteration)
    curvePoint rocArea(curvePoint l, curvePoint r) := TRANSFORM
      deltaPos  := if(l.tpr > r.tpr, l.tpr - r.tpr, 0.0);
      deltaNeg  := if( l.fpr > r.fpr, l.fpr - r.fpr, 0.0);
      SELF.deltaPos := deltaPos;
      SELF.deltaNeg := deltaNeg;
      // A classification without incorrectly classified instances must return AUC = 1
      SELF.AUC      := IF(r.fpr=0 AND l.tpr=0 AND r.tpr=1, 1, l.AUC) + deltaPos * (l.cumNeg + 0.5* deltaNeg);
      SELF.cumNeg   := l.cumNeg + deltaNeg;
      SELF:= r;
    END;
    RETURN ITERATE(rocPoints, rocArea(LEFT, RIGHT));
  END;
