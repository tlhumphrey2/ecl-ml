 IMPORT ML;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

 // Support Vector Machine.
  //  see https://en.wikipedia.org/wiki/Support_vector_machine
  // Use the SVM attributes directly for scaling and grid search.  This
  //module acts as a facade to the actual SVM attributes and provides only
  //the interface and capabilities defined for the Classify abstract.
  //
  // The inputs are:
  /*  svm_type : set type of SVM, SVM.Types.SVM_Type enum
              C_SVC    (multi-class classification)
              NU_SVC   (multi-class classification)
              ONE_CLASS SVM
              EPSILON_SVR  (regression)
              NU_SVR   (regression)
      kernel_type : set type of kernel function, SVM.Types.Kernel_Type enum
              LINEAR: u'*v
              POLY:   polynomial,  (gamma*u'*v + coef0)^degree
              RBF:    radial basis function: exp(-gamma*|u-v|^2)
              SIGMOID: tanh(gamma*u'*v + coef0)
              PRECOMPUTED: precomputed kernel (kernel values in training_set_file)
      degree : degree in kernel function for POLY
      gamma  : gamma in kernel function for POLY, RBF, and SIGMOID
      coef0  : coef0 in kernel function for POLY, SIGMOID
      cost   : the parameter C of C-SVC, epsilon-SVR, and nu-SVR
      eps    : the epsilon for stopping
      nu     : the parameter nu of nu-SVC, one-class SVM, and nu-SVR
      p      : the epsilon in loss function of epsilon-SVR
      shrinking : whether to use the shrinking heuristics, default TRUE
  */
  // The LibSVM development package must be installed on your cluster!
  EXPORT SVM(SVM.Types.SVM_Type svm_type, SVM.Types.Kernel_Type kernel_type,
             INTEGER4 degree, REAL8 gamma, REAL8 coef0, REAL8 cost, REAL8 eps,
             REAL8 nu, REAL8 p, BOOLEAN shrinking) := MODULE(DEFAULT)
    SVM.Types.Training_Parameters
    makeParm(UNSIGNED4 dep_field, SVM.Types.SVM_Type svm_type,
             SVM.Types.Kernel_Type kernel_type,
             INTEGER4 degree, REAL8 gamma, REAL8 coef0, REAL8 cost,
             REAL8 eps, REAL8 nu, REAL8 p, BOOLEAN shrinking) := TRANSFORM
      SELF.id := dep_field;
      SELF.svmType := svm_type;
      SELF.kernelType := kernel_type;
      SELF.degree := degree;
      SELF.gamma := gamma;
      SELF.coef0 := coef0;
      SELF.C := cost;
      SELF.eps := eps;
      SELF.nu := nu;
      SELF.p := p;
      SELF.shrinking := shrinking;
      SELF.prob_est := FALSE;
      SELF := [];
    END;
    SHARED Training_Param(UNSIGNED4 df) := ROW(makeParm(df, svm_type,
                                          kernel_type, degree,
                                          gamma, coef0, cost, eps, nu,
                                          p, shrinking));
    // Learn from continuous data
    EXPORT LearnC(DATASET(Types.NumericField) Indep,
                  DATASET(Types.DiscreteField) Dep) := FUNCTION
      depc := PROJECT(Dep, Types.NumericField);
      inst_data := SVM.Converted.ToInstance(Indep, Depc);
      dep_field := dep[1].number;
      tp := DATASET(Training_Param(dep_field));
      mdl := SVM.train(tp, inst_data);
      nf_mdl := SVM.Converted.FromModel(Base, mdl);
      RETURN nf_mdl; // All classifiers serialized to numeric field format
    END;
    // Learn from discrete data uses DEFAULT implementation
    // Classify continuous data - using a prebuilt model
    EXPORT ClassifyC(DATASET(Types.NumericField) Indep,
                     DATASET(Types.NumericField) mod) := FUNCTION
      inst_data := SVM.Converted.ToInstance(Indep);
      mdl := SVM.Converted.ToModel(mod);
      pred := SVM.predict(mdl, inst_data).Prediction;
      // convert to standard form
      l_result cvt(SVM.Types.SVM_Prediction p) := TRANSFORM
        SELF.id := p.rid;
        SELF.number := p.id; // model ID is the dependent var field ID
        SELF.value := p.predict_y;
        SELF.conf := 0.5; // no confidence measures
      END;
      rslt := PROJECT(pred, cvt(LEFT));
      RETURN rslt;
    END;
    // Classify discrete data - uses DEFAULT implementation
  END; // SVM
