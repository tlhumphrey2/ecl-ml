 IMPORT ML;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

  /*
  Apply classification with Neural Network by using NeuralNetworks.ecl
  */
  EXPORT NeuralNetworksClassifier (DATASET(Types.DiscreteField) net, DATASET(Mat.Types.MUElement) IntW, DATASET(Mat.Types.MUElement) Intb, REAL8 LAMBDA=0.001, REAL8 ALPHA=0.1, UNSIGNED2 MaxIter=100, 
  UNSIGNED4 prows=0, UNSIGNED4 pcols=0, UNSIGNED4 Maxrows=0, UNSIGNED4 Maxcols=0) := MODULE(DEFAULT)
  SHARED NN := NeuralNetworks(net, prows,  pcols, Maxrows,  Maxcols);
  EXPORT LearnC(DATASET(Types.NumericField) Indep, DATASET(Types.DiscreteField) Dep) := FUNCTION
    Y := PROJECT(Dep,Types.NumericField);
    groundTruth:= Utils.ToGroundTruth (Y);//groundTruth is a matrix that each column correspond to one sample
    //to convert the groundTruth matrix to NumericFiled format firt I have to trasnpose it to make each sample to correspond to
    //each row, in that case when we convert it to NumericFiled format the id filed is built up correctly
    groundTruth_t := Mat.trans(groundTruth);
    groundTruth_NumericField := Types.FromMatrix (groundTruth_t);
    Learntmodel := NN.NNLearn(Indep, groundTruth_NumericField,IntW, Intb,  LAMBDA, ALPHA, MaxIter);
    RETURN Learntmodel;
  END;
  EXPORT Model(DATASET(Types.NumericField) Lmod) := FUNCTION
    RETURN NN.Model(Lmod);
  END;
  EXPORT ClassProbDistribC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) :=FUNCTION
    AEnd := NN.NNOutput(Indep,mod);
    RETURN AEnd;
  END;
  EXPORT ClassifyC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
    Classes := NN.NNClassify(Indep,mod);
    RETURN Classes;
  END;
  END;//END NeuralNetworksClassifier
