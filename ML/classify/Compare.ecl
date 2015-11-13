 IMPORT ML;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

/*
// Function to compute the efficacy of a given classification process
// Expects the dependents (classification tags deemed to be true)
// Computeds - classification tags created by the classifier
*/
EXPORT Compare(DATASET(Types.DiscreteField) Dep,DATASET(l_result) Computed) := MODULE
  DiffRec := RECORD
    Types.t_FieldNumber classifier;  // The classifier in question (value of 'number' on outcome data)
    Types.t_Discrete  c_actual;      // The value of c provided
    Types.t_Discrete  c_modeled;     // The value produced by the classifier
    Types.t_FieldReal score;         // Score allocated by classifier
  END;
  DiffRec  notediff(Computed le,Dep ri) := TRANSFORM
    SELF.c_actual := ri.value;
    SELF.c_modeled := le.value;
    SELF.score := le.conf;
    SELF.classifier := ri.number;
  END;
  SHARED J := JOIN(Computed,Dep,LEFT.id=RIGHT.id AND LEFT.number=RIGHT.number,notediff(LEFT,RIGHT));
  // Building the Confusion Matrix
  SHARED ConfMatrix_Rec := RECORD
    Types.t_FieldNumber classifier; // The classifier in question (value of 'number' on outcome data)
    Types.t_Discrete c_actual;      // The value of c provided
    Types.t_Discrete c_modeled;     // The value produced by the classifier
    Types.t_FieldNumber Cnt:=0;     // Number of occurences
  END;
  SHARED class_cnt := TABLE(Dep,{classifier:= number, c_actual:= value, Cnt:= COUNT(GROUP)},number, value, FEW); // Looking for class values
  ConfMatrix_Rec form_cfmx(class_cnt le, class_cnt ri) := TRANSFORM
    SELF.classifier := le.classifier;
    SELF.c_actual:= le.c_actual;
    SELF.c_modeled:= ri.c_actual;
  END;
  SHARED cfmx := JOIN(class_cnt, class_cnt, LEFT.classifier = RIGHT.classifier, form_cfmx(LEFT, RIGHT)); // Initialzing the Confusion Matrix with 0 counts
  SHARED cross_raw := TABLE(J,{classifier,c_actual,c_modeled,Cnt := COUNT(GROUP)},classifier,c_actual,c_modeled,FEW); // Counting ocurrences
  ConfMatrix_Rec form_confmatrix(ConfMatrix_Rec le, cross_raw ri) := TRANSFORM
    SELF.Cnt  := ri.Cnt;
    SELF      := le;
  END;
//CrossAssignments, it returns information about actual and predicted classifications done by a classifier
//                  also known as Confusion Matrix
  EXPORT CrossAssignments := JOIN(cfmx, cross_raw,
                              LEFT.classifier = RIGHT.classifier AND LEFT.c_actual = RIGHT.c_actual AND LEFT.c_modeled = RIGHT.c_modeled,
                              form_confmatrix(LEFT,RIGHT), LEFT OUTER, LOOKUP);
//RecallByClass, it returns the proportion of instances belonging to a class that was correctly classified,
//               also know as True positive rate and sensivity, TP/(TP+FN).
  EXPORT RecallByClass := SORT(TABLE(CrossAssignments, {classifier, c_actual, tp_rate := SUM(GROUP,IF(c_actual=c_modeled,cnt,0))/SUM(GROUP,cnt)}, classifier, c_actual, FEW), classifier, c_actual);
//PrecisionByClass, returns the proportion of instances classified as a class that really belong to this class: TP /(TP + FP).
  EXPORT PrecisionByClass := SORT(TABLE(CrossAssignments,{classifier,c_modeled, Precision := SUM(GROUP,IF(c_actual=c_modeled,cnt,0))/SUM(GROUP,cnt)},classifier,c_modeled,FEW), classifier, c_modeled);
//FP_Rate_ByClass, it returns the proportion of instances not belonging to a class that were incorrectly classified as this class,
//                 also known as False Positive rate FP / (FP + TN).
  FalseRate_rec := RECORD
    Types.t_FieldNumber classifier; // The classifier in question (value of 'number' on outcome data)
    Types.t_Discrete c_modeled;     // The value produced by the classifier
    Types.t_FieldReal fp_rate;      // False Positive Rate
  END;
  wrong_modeled:= TABLE(CrossAssignments(c_modeled<>c_actual), {classifier, c_modeled, wcnt:= SUM(GROUP, cnt)}, classifier, c_modeled);
  j2:= JOIN(wrong_modeled, class_cnt, LEFT.classifier=RIGHT.classifier AND LEFT.c_modeled<>RIGHT.c_actual);
  allfalse:= TABLE(j2, {classifier, c_modeled, not_actual:= SUM(GROUP, cnt)}, classifier, c_modeled);
  EXPORT FP_Rate_ByClass := JOIN(wrong_modeled, allfalse, LEFT.classifier=RIGHT.classifier AND LEFT.c_modeled=RIGHT.c_modeled,
                          TRANSFORM(FalseRate_rec, SELF.fp_rate:= LEFT.wcnt/RIGHT.not_actual, SELF:= LEFT));
// Accuracy, it returns the proportion of instances correctly classified (total, without class distinction)
  EXPORT Accuracy := TABLE(CrossAssignments, {classifier, Accuracy:= SUM(GROUP,IF(c_actual=c_modeled,cnt,0))/SUM(GROUP, cnt)}, classifier);
END;
