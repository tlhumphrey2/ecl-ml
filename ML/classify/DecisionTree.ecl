 IMPORT ML;
IMPORT * FROM $;
IMPORT * FROM ML;
//IMPORT $.Mat;
IMPORT * FROM ML.Types;
IMPORT PBblas;
IMPORT ML.SVM;
Layout_Cell := PBblas.Types.Layout_Cell;

/* From Wikipedia: 
http://en.wikipedia.org/wiki/Decision_tree_learning#General
"... Decision tree learning is a method commonly used in data mining.
The goal is to create a model that predicts the value of a target variable based on several input variables.
... A tree can be "learned" by splitting the source set into subsets based on an attribute value test. 
This process is repeated on each derived subset in a recursive manner called recursive partitioning. 
The recursion is completed when the subset at a node has all the same value of the target variable,
or when splitting no longer adds value to the predictions.
This process of top-down induction of decision trees (TDIDT) [1] is an example of a greedy algorithm,
and it is by far the most common strategy for learning decision trees from data, but it is not the only strategy."
The module can learn using different splitting algorithms, and return a model.
The Decision Tree (model) has the same structure independently of which split algorithm was used.
The model  is used to predict the class from new examples.
*/
  EXPORT DecisionTree := MODULE
/*  
    Decision Tree Learning using Gini Impurity-Based criterion
*/
    EXPORT GiniImpurityBased(INTEGER1 Depth=10, REAL Purity=1.0):= MODULE(DEFAULT)
      EXPORT LearnD(DATASET(Types.DiscreteField) Indep, DATASET(Types.DiscreteField) Dep) := FUNCTION
        nodes := ML.Trees.SplitsGiniImpurBased(Indep, Dep, Depth, Purity);
        RETURN ML.Trees.ToDiscreteTree(nodes);
      END;
      EXPORT ClassProbDistribD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod) :=FUNCTION
        RETURN ML.Trees.ClassProbDistribD(Indep, mod);
      END;
      EXPORT ClassifyD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
        RETURN ML.Trees.ClassifyD(Indep,mod);
      END;
      EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
        RETURN ML.Trees.ModelD(mod);
      END;
    END;  // Gini Impurity DT Module
/*
    Decision Tree using C4.5 Algorithm (Quinlan, 1987)
*/
    EXPORT C45(BOOLEAN Pruned= TRUE, INTEGER1 numFolds = 3, REAL z = 0.67449) := MODULE(DEFAULT)
      EXPORT LearnD(DATASET(Types.DiscreteField) Indep, DATASET(Types.DiscreteField) Dep) := FUNCTION
        nodes := IF(Pruned, Trees.SplitsIGR_Pruned(Indep, Dep, numFolds, z), Trees.SplitsInfoGainRatioBased(Indep, Dep));
        RETURN ML.Trees.ToDiscreteTree(nodes);
      END;
      EXPORT ClassProbDistribD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod) :=FUNCTION
        RETURN ML.Trees.ClassProbDistribD(Indep, mod);
      END;
      EXPORT ClassifyD(DATASET(Types.DiscreteField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
        RETURN ML.Trees.ClassifyD(Indep,mod);
      END;
      EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
        RETURN ML.Trees.ModelD(mod);
      END;
    END;  // C45 DT Module

/*  C45 Binary Decision Tree
    It learns from continuous data and builds a Binary Decision Tree based on Info Gain Ratio
    Configuration Input
      minNumObj   minimum number of instances in a leaf node, used in splitting process
      maxLevel    stop learning criteria, either tree's level reachs maxLevel depth or no more split can be done.
*/
    EXPORT C45Binary(t_Count minNumObj=2, ML.Trees.t_level maxLevel=32) := MODULE(DEFAULT)
      EXPORT LearnC(DATASET(Types.NumericField) Indep, DATASET(Types.DiscreteField) Dep) := FUNCTION
        nodes := Trees.SplitBinaryCBased(Indep, Dep, minNumObj, maxLevel);
        RETURN ML.Trees.ToNumericTree(nodes);
      END;
      EXPORT ClassifyC(DATASET(Types.NumericField) Indep,DATASET(Types.NumericField) mod) := FUNCTION
        RETURN ML.Trees.ClassifyC(Indep,mod);
      END;
      EXPORT Model(DATASET(Types.NumericField) mod) := FUNCTION
        RETURN ML.Trees.ModelC(mod);
      END;
    END; // C45Binary DT Module
  END; // DecisionTree Module
