 IMPORT ML;
IMPORT * FROM ML;
IMPORT * FROM $;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

/* From http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#overview
   "... Random Forests grows many classification trees.
   To classify a new object from an input vector, put the input vector down each of the trees in the forest.
   Each tree gives a classification, and we say the tree "votes" for that class.
   The forest chooses the classification having the most votes (over all the trees in the forest).

   Each tree is grown as follows:
   - If the number of cases in the training set is N, sample N cases at random - but with replacement, from the original data.
     This sample will be the training set for growing the tree.
   - If there are M input variables, a number m<<M is specified such that at each node, m variables are selected at random out of the M
     and the best split on these m is used to split the node. The value of m is held constant during the forest growing.
   - Each tree is grown to the largest extent possible. There is no pruning. ..."

Configuration Input
   treeNum    number of trees to generate
   fsNum      number of features to sample each iteration
   Purity     p <= 1.0
   Depth      max tree level
*/
  EXPORT RandomForest(t_Count treeNum, t_Count fsNum, REAL Purity=1.0, INTEGER1 Depth=32):= MODULE
    EXPORT LearnD(DATASET(Types.DiscreteField) Indep, DATASET(Types.DiscreteField) Dep) := FUNCTION
      nodes := Ensemble.SplitFeatureSampleGI(Indep, Dep, treeNum, fsNum, Purity, Depth);
      RETURN ML.Ensemble.ToDiscreteForest(nodes);
    END;
    EXPORT LearnC(DATASET(Types.NumericField) Indep, DATASET(Types.DiscreteField) Dep) := FUNCTION
      nodes := Ensemble.SplitFeatureSampleGIBin(Indep, Dep, treeNum, fsNum, Purity, Depth);
      RETURN ML.Ensemble.ToContinuosForest(nodes);
    END;
    // Transform NumericFiled "mod" to Ensemble.gSplitF "discrete tree nodes" model format using field map model_Map
    EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
      RETURN ML.Ensemble.FromDiscreteForest(mod);
    END;
    // Transform NumericFiled "mod" to Ensemble.gSplitC "binary tree nodes" model format using field map modelC_Map
    EXPORT ModelC(DATASET(Types.NumericField) mod) := FUNCTION
      RETURN ML.Ensemble.FromContinuosForest(mod);
    END;
    // The functions return instances' class probability distribution for each class value
    // based upon independent values (Indep) and the ensemble model (mod).
    EXPORT ClassProbDistribD(DATASET(Types.DiscreteField) Indep, DATASET(Types.NumericField) mod) :=FUNCTION
      RETURN ML.Ensemble.ClassProbDistribForestD(Indep, mod);
    END;
    EXPORT ClassProbDistribC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) :=FUNCTION
      RETURN ML.Ensemble.ClassProbDistribForestC(Indep, mod);
    END;
    // Classification functions based upon independent values (Indep) and the ensemble model (mod).
    EXPORT ClassifyD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
      RETURN ML.Ensemble.ClassifyDForest(Indep, mod);
    END;
    EXPORT ClassifyC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
      RETURN ML.Ensemble.ClassifyCForest(Indep,mod);
    END;
  END; // RandomForest module
