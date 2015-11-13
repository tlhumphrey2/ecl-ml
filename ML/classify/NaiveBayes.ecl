 IMPORT ML;
IMPORT * FROM ML;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

  EXPORT NaiveBayes := MODULE(DEFAULT)
    SHARED SampleCorrection := 1;
    SHARED LogScale(REAL P) := -LOG(P)/LOG(2);

/* Naive Bayes Classification 
   This method can support producing classification results for multiple classifiers at once
   Note the presumption that the result (a weight for each value of each field) can fit in memory at once
*/
    SHARED BayesResult := RECORD
      Types.t_RecordId    id := 0;        // A record-id - allows a model to have an ordered sequence of results
      Types.t_Discrete    class_number;   // Dependent "number" value - Classifier ID
      Types.t_discrete    c;              // Dependent "value" value - Class value
      Types.t_FieldNumber number;         // A reference to a feature (or field) in the independants
      Types.t_Count       Support;        // Number of cases
    END;
    SHARED BayesResultD := RECORD (BayesResult)
      Types.t_discrete  f := 0;           // Independant value - Attribute value
      Types.t_FieldReal PC;                // Either P(F|C) or P(C) if number = 0. Stored in -Log2(P) - so small is good :)
    END;
    SHARED BayesResultC := RECORD (BayesResult)
      Types.t_FieldReal  mu:= 0;          // Independent attribute mean (mu)
      Types.t_FieldReal  var:= 0;         // Independent attribute sample standard deviation (sigma squared)
    END;
/*
  The inputs to the BuildNaiveBayes are:
  a) A dataset of discretized independant variables
  b) A dataset of class results (these must match in ID the discretized independant variables).
     Some routines can produce multiple classifiers at once; if so these are distinguished using the NUMBER field of cl
*/
    EXPORT LearnD(DATASET(Types.DiscreteField) Indep,DATASET(Types.DiscreteField) Dep) := FUNCTION
      dd := Indep;
      cl := Dep;
      Triple := RECORD
        Types.t_Discrete c;
        Types.t_Discrete f;
        Types.t_FieldNumber number;
        Types.t_FieldNumber class_number;
      END;
      Triple form(dd le,cl ri) := TRANSFORM
        SELF.c := ri.value;
        SELF.f := le.value;
        SELF.number := le.number;
        SELF.class_number := ri.number;
      END;
      Vals := JOIN(dd,cl,LEFT.id=RIGHT.id,form(LEFT,RIGHT));
      AggregatedTriple := RECORD
        Vals.c;
        Vals.f;
        Vals.number;
        Vals.class_number;
        Types.t_Count support := COUNT(GROUP);
      END;
      // This is the raw table - how many of each value 'f' for each field 'number' appear for each value 'c' of each classifier 'class_number'
      Cnts := TABLE(Vals,AggregatedTriple,c,f,number,class_number,FEW);
      // Compute P(C)
      CTots := TABLE(cl,{value,number,Support := COUNT(GROUP)},value,number,FEW);
      CLTots := TABLE(CTots,{number,TSupport := SUM(GROUP,Support), GC := COUNT(GROUP)},number,FEW);
      P_C_Rec := RECORD
        Types.t_Discrete c;            // The value within the class
        Types.t_Discrete f;            // The number of features within the class
        Types.t_Discrete class_number; // Used when multiple classifiers being produced at once
        Types.t_FieldReal support;     // Used to store total number of C
        REAL8 w;                       // P(C)
      END;
      // Apply Laplace Estimator to P(C) in order to be consistent with attributes' probability
      P_C_Rec pct(CTots le,CLTots ri) := TRANSFORM
        SELF.c := le.value;
        SELF.f := 0; // to be claculated later on
        SELF.class_number := ri.number;
        SELF.support := le.Support + SampleCorrection;
        SELF.w := (le.Support + SampleCorrection) / (ri.TSupport + ri.GC*SampleCorrection);
      END;
      PC_0 := JOIN(CTots,CLTots,LEFT.number=RIGHT.number,pct(LEFT,RIGHT),FEW);
      // Computing Attributes' probability
      AttribValue_Rec := RECORD
        Cnts.class_number;  // Used when multiple classifiers being produced at once
        Cnts.number;        // A reference to a feature (or field) in the independants
        Cnts.f;             // Independant value
        Types.t_Count support := 0;
      END;
      // Generating feature domain per feature (class_number only used when multiple classifiers being produced at once)
      AttValues := TABLE(Cnts, AttribValue_Rec, class_number, number, f, FEW);
      AttCnts   := TABLE(AttValues, {class_number, number, ocurrence:= COUNT(GROUP)},class_number, number, FEW); // Summarize 
      AttrValue_per_Class_Rec := RECORD
        Types.t_Discrete c;
        AttValues.f;
        AttValues.number;
        AttValues.class_number;
        AttValues.support;
      END;
      // Generating class x feature domain, initial support = 0
      AttrValue_per_Class_Rec form_cl_attr(AttValues le, CTots ri):= TRANSFORM
        SELF.c:= ri.value;
        SELF:= le;
      END;
      ATots:= JOIN(AttValues, CTots, LEFT.class_number = RIGHT.number, form_cl_attr(LEFT, RIGHT), MANY LOOKUP, FEW);
      // Counting feature value ocurrence per class x feature, remains 0 if combination not present in dataset
      ATots form_ACnts(ATots le, Cnts ri) := TRANSFORM
        SELF.support  := ri.support;
        SELF      := le;
      END;
      ACnts := JOIN(ATots, Cnts, LEFT.c = RIGHT.c AND LEFT.f = RIGHT.f AND LEFT.number = RIGHT.number AND LEFT.class_number = RIGHT.class_number, 
                            form_ACnts(LEFT,RIGHT),
                            LEFT OUTER, LOOKUP);
      // Summarizing and getting value 'GC' to apply in Laplace Estimator
      TotalFs0 := TABLE(ACnts,{c,number,class_number,Types.t_Count Support := SUM(GROUP,Support),GC := COUNT(GROUP)},c,number,class_number,FEW);
      TotalFs := TABLE(TotalFs0,{c,class_number,ML.Types.t_Count Support := SUM(GROUP,Support),Types.t_Count GC := SUM(GROUP,GC)},c,class_number,FEW);
      // Merge and Laplace Estimator
      F_Given_C_Rec := RECORD
        ACnts.c;
        ACnts.f;
        ACnts.number;
        ACnts.class_number;
        ACnts.support;
        REAL8 w;
      END;
      F_Given_C_Rec mp(ACnts le,TotalFs ri) := TRANSFORM
        SELF.support := le.Support+SampleCorrection;
        SELF.w := (le.Support+SampleCorrection) / (ri.Support+ri.GC*SampleCorrection);
        SELF := le;
      END;
      // Calculating final probabilties
      FC := JOIN(ACnts,TotalFs,LEFT.C = RIGHT.C AND LEFT.class_number=RIGHT.class_number,mp(LEFT,RIGHT),LOOKUP);
      PC_0 form_TotalFs(PC_0 le, TotalFs ri) := TRANSFORM
        SELF.f  := ri.Support+ri.GC*SampleCorrection;
        SELF    := le;
      END;
      PC := JOIN(PC_0, TotalFs, LEFT.C = RIGHT.C AND LEFT.class_number=RIGHT.class_number,form_TotalFs(LEFT,RIGHT),LOOKUP);   
      Pret := PROJECT(FC,TRANSFORM(BayesResultD, SELF.PC:=LEFT.w, SELF := LEFT))+PROJECT(PC,TRANSFORM(BayesResultD, SELF.PC:=LEFT.w, SELF.number:= 0,SELF:=LEFT));
      Pret1 := PROJECT(Pret,TRANSFORM(BayesResultD, SELF.PC := LogScale(LEFT.PC),SELF.id := Base+COUNTER,SELF := LEFT));
      ToField(Pret1,o);
      RETURN o;
    END;
    // Transform NumericFiled "mod" to discrete Naive Bayes format model "BayesResultD"
    EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
      ML.FromField(mod,BayesResultD,o);
      RETURN o;
    END;
    // This function will take a pre-existing NaiveBayes model (mo) and score every row of a discretized dataset
    // The output will have a row for every row of dd and a column for every class in the original training set
    EXPORT ClassProbDistribD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
      d := Indep;
      mo := Model(mod);
      // Firstly we can just compute the support for each class from the bayes result
      dd := DISTRIBUTE(d,HASH(id)); // One of those rather nice embarassingly parallel activities
      Inter := RECORD
        Types.t_discrete c;
        Types.t_discrete class_number;
        Types.t_RecordId Id;
        REAL8  w;
      END;
      Inter note(dd le,mo ri) := TRANSFORM
        SELF.c := ri.c;
        SELF.class_number := ri.class_number;
        SELF.id := le.id;
        SELF.w := ri.PC;
      END;
  // RHS is small so ,ALL join should work ok
  // Ignore the "explicitly distributed" compiler warning - the many lookup is preserving the distribution
      J := JOIN(dd,mo,LEFT.number=RIGHT.number AND LEFT.value=RIGHT.f,note(LEFT,RIGHT),MANY LOOKUP);
      InterCounted := RECORD
        J.c;
        J.class_number;
        J.id;
        REAL8 P := SUM(GROUP,J.W);
        Types.t_FieldNumber Missing := COUNT(GROUP); // not really missing just yet :)
      END;
      TSum := TABLE(J,InterCounted,c,class_number,id,LOCAL);
  // Now we have the sums for all the F present for each class we need to
  // a) Add in the P(C)
  // b) Suitably penalize any 'f' which simply were not present in the model
  // We start by counting how many not present ...
      FTots := TABLE(DD,{id,c := COUNT(GROUP)},id,LOCAL);
      InterCounted NoteMissing(TSum le,FTots ri) := TRANSFORM
        SELF.Missing := ri.c - le.Missing;
        SELF := le;
      END;
      MissingNoted := JOIN(Tsum,FTots,LEFT.id=RIGHT.id,NoteMissing(LEFT,RIGHT),LOOKUP);
      InterCounted NoteC(MissingNoted le,mo ri) := TRANSFORM
        SELF.P := le.P+ri.PC+le.Missing*LogScale(SampleCorrection/ri.f);
        SELF := le;
      END;
      CNoted := JOIN(MissingNoted,mo(number=0),LEFT.c=RIGHT.c,NoteC(LEFT,RIGHT),LOOKUP);
      l_result toResult(CNoted le) := TRANSFORM
        SELF.id := le.id;               // Instance ID
        SELF.number := le.class_number; // Classifier ID
        SELF.value := le.c;             // Class value
        SELF.conf := POWER(2.0, -le.p); // Convert likehood to decimal value
      END;
      // Normalizing Likehood to deliver Class Probability per instance
      InstResults := PROJECT(CNoted, toResult(LEFT), LOCAL);
      gInst := TABLE(InstResults, {number, id, tot:=SUM(GROUP,conf)}, number, id, LOCAL);
      clDist:= JOIN(InstResults, gInst,LEFT.number=RIGHT.number AND LEFT.id=RIGHT.id, TRANSFORM(Types.l_result, SELF.conf:=LEFT.conf/RIGHT.tot, SELF:=LEFT), LOCAL);
      RETURN clDist;
    END;
    // Classification function for discrete independent values and model
    EXPORT ClassifyD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
      // get class probabilities for each instance
      dClass:= ClassProbDistribD(Indep, mod);
      // select the class with greatest probability for each instance
      sClass := SORT(dClass, id, -conf, LOCAL);
      finalClass:=DEDUP(sClass, id, LOCAL);
      RETURN finalClass;
    END;
    /*From Wikipedia
    " ...When dealing with continuous data, a typical assumption is that the continuous values associated with each class are distributed according to a Gaussian distribution.
    For example, suppose the training data contain a continuous attribute, x. We first segment the data by the class, and then compute the mean and variance of x in each class.
    Let mu_c be the mean of the values in x associated with class c, and let sigma^2_c be the variance of the values in x associated with class c.
    Then, the probability density of some value given a class, P(x=v|c), can be computed by plugging v into the equation for a Normal distribution parameterized by mu_c and sigma^2_c..."
    */
    EXPORT LearnC(DATASET(NumericField) Indep, DATASET(DiscreteField) Dep) := FUNCTION
      Triple := RECORD
        Types.t_FieldNumber class_number;
        Types.t_FieldNumber number;
        Types.t_FieldReal value;
        Types.t_Discrete c;
      END;
      Triple form(Indep le, Dep ri) := TRANSFORM
        SELF.class_number := ri.number;
        SELF.number := le.number;
        SELF.value := le.value;
        SELF.c := ri.value;
      END;
      Vals := JOIN(Indep, Dep, LEFT.id=RIGHT.id, form(LEFT,RIGHT));
      // Compute P(C)
      ClassCnts := TABLE(Dep, {number, value, support := COUNT(GROUP)}, number, value, FEW);
      ClassTots := TABLE(ClassCnts,{number, TSupport := SUM(GROUP,Support)}, number, FEW);
      P_C_Rec := RECORD
        Types.t_Discrete class_number; // Used when multiple classifiers being produced at once
        Types.t_Discrete c;             // The class value "C"
        Types.t_FieldReal support;          // Cases count
        Types.t_FieldReal  mu:= 0;          // P(C)
      END;
      // Computing prior probability P(C)
      P_C_Rec pct(ClassCnts le, ClassTots ri) := TRANSFORM
        SELF.class_number := ri.number;
        SELF.c := le.value;
        SELF.support := le.Support;
        SELF.mu := le.Support/ri.TSupport;
      END;
      PC := JOIN(ClassCnts, ClassTots, LEFT.number=RIGHT.number, pct(LEFT,RIGHT), FEW);
      PC_cnt := COUNT(PC);
      // Computing Attributes' mean and variance. mu_c and sigma^2_c.
      AggregatedTriple := RECORD
        Vals.class_number;
        Vals.c;
        Vals.number;
        Types.t_Count support := COUNT(GROUP);
        Types.t_FieldReal mu:=AVE(GROUP, Vals.value);
        Types.t_FieldReal var:= VARIANCE(GROUP, Vals.value);
      END;
      AC:= TABLE(Vals, AggregatedTriple, class_number, c, number);
      Pret := PROJECT(PC, TRANSFORM(BayesResultC, SELF.id := Base + COUNTER, SELF.number := 0, SELF:=LEFT)) +
              PROJECT(AC, TRANSFORM(BayesResultC, SELF.id := Base + COUNTER + PC_cnt, SELF.var:= LEFT.var*LEFT.support/(LEFT.support -1), SELF := LEFT));
      ToField(Pret,o);
      RETURN o;
    END;
    // Transform NumericFiled "mod" to continuos Naive Bayes format model "BayesResultC"
    EXPORT ModelC(DATASET(Types.NumericField) mod) := FUNCTION
      ML.FromField(mod,BayesResultC,o);
      RETURN o;
    END;
    EXPORT ClassProbDistribC(DATASET(Types.NumericField) Indep, DATASET(Types.NumericField) mod) := FUNCTION
      dd := DISTRIBUTE(Indep, HASH(id));
      mo := ModelC(mod);
      Inter := RECORD
        Types.t_FieldNumber class_number;
        Types.t_FieldNumber number;
        Types.t_FieldReal value;
        Types.t_Discrete c;
        Types.t_RecordId Id;
        Types.t_FieldReal  likehood:=0; // Probability density P(x=v|c)
      END;
      Inter ProbDensity(dd le, mo ri) := TRANSFORM
        SELF.id := le.id;
        SELF.value:= le.value;
        SELF.likehood := LogScale(exp(-(le.value-ri.mu)*(le.value-ri.mu)/(2*ri.var))/SQRT(2*ML.Utils.Pi*ri.var));
        SELF:= ri;
      END;
      // Likehood or probability density P(x=v|c) is calculated assuming Gaussian distribution of the class based on new instance attribute value and atribute's mean and variance from model
      LogPall := JOIN(dd,mo,LEFT.number=RIGHT.number , ProbDensity(LEFT,RIGHT),MANY LOOKUP);
      // Prior probaility PC
      LogPC:= PROJECT(mo(number=0),TRANSFORM(BayesResultC, SELF.mu:=LogScale(LEFT.mu), SELF:=LEFT));
      post_rec:= RECORD
        LogPall.id;
        LogPall.class_number;
        LogPall.c;
        Types.t_FieldReal prod:= SUM(GROUP, LogPall.likehood);
      END;
      // Likehood and Prior are expressed in LogScale, summing really means multiply
      LikehoodProduct:= TABLE(LogPall, post_rec, class_number, c, id, LOCAL);
      // Posterior probability = prior x likehood_product / evidence
      // We use only the numerator of that fraction, because the denominator is effectively constant.
      // See: http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model
      l_result toResult(LikehoodProduct le, LogPC ri) := TRANSFORM
        SELF.id := le.id;               // Instance ID
        SELF.number := le.class_number; // Classifier ID
        SELF.value := ri.c;             // Class value
        SELF.conf:= le.prod + ri.mu;    // Adding mu
      END;
      AllPosterior:= JOIN(LikehoodProduct, LogPC, LEFT.class_number = RIGHT.class_number AND LEFT.c = RIGHT.c, toResult(LEFT, RIGHT), LOOKUP);
      // Normalizing Likehood to deliver Class Probability per instance
      baseExp:= TABLE(AllPosterior, {id, minConf:= MIN(GROUP, conf)},id, LOCAL); // will use this to divide instance's conf by the smallest per id
      l_result toNorm(AllPosterior le, baseExp ri) := TRANSFORM
        SELF.conf:= POWER(2.0, -MIN( le.conf - ri.minConf, 2048));  // minimum probability set to 1/2^2048 = 0 at the end
        SELF:= le;
      END;
      AllOffset:= JOIN(AllPosterior, baseExp, LEFT.id = RIGHT.id, toNorm(LEFT, RIGHT), LOOKUP); // at least one record per id with 1.0 probability before normalization
      gInst := TABLE(AllOffset, {number, id, tot:=SUM(GROUP,conf)}, number, id, LOCAL);
      clDist:= JOIN(AllOffset, gInst,LEFT.number=RIGHT.number AND LEFT.id=RIGHT.id, TRANSFORM(Types.l_result, SELF.conf:=LEFT.conf/RIGHT.tot, SELF:=LEFT), LOCAL);
      RETURN clDist;
    END;
    // Classification function for continuous independent values and model
    EXPORT ClassifyC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
      // get class probabilities for each instance
      dClass:= ClassProbDistribC(Indep, mod);
      // select the class with greatest probability for each instance
      sClass := SORT(dClass, id, -conf, LOCAL);
      finalClass:=DEDUP(sClass, id, LOCAL);
      RETURN finalClass;
     END;
  END; // NaiveBayes Module
