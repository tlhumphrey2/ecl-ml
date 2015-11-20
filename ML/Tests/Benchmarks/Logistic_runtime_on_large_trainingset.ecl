IMPORT * FROM ML;
IMPORT ML10;
/*
This code uses data gotten online at https://archive.ics.uci.edu/ml/datasets/Covertype. This code ONLY assumes
the data file was uploaded to the landing zone and then sprayed to the nodes of the THOR. Here is what this
code does:
1. Converts the fields of the dataset from string to the field types of the covtype_rec record layout(below).
2. Makes the dataset suitable for Logistic by making sure the dependent variable/field, CoverType, has only 2 
   values: 0 or 1. It does this by changing CoverType to 1 if the original record had CoverType=2 and CoverType is 0
   if the original record had CoverType as something other than 2 (This means the model produced can distingish 
   between Forest Cover Type 2 sand other Cover types.
3. Separates the independent variables/fields from the dependent variables/fields
4. Converts both the independent and dependent datasets to the record format used by ML, i.e. ML.Types.NumericField
5. Does Logistic Regression learning to produce a model and OUTPUTs the model to the workunit.

To make the execution time of this test code reflect the time it took to learn the model, put tasks 1 through 4 in
another file and execute it. Have it produce a NumericField dataset which would be read in before doing task 5.

EXECUTION TIME OF THIS CODE: It took 3 minutes and 9 seconds on a 20 node THOR. Here are the specs of the 
hardware:

CPU = Intel(R) Xeon(TM) CPU 3.20GHz
RAM = 4GB DDR2
NIC = 1Gbit

*/

//-------------------------------------------------------------------------------------------------
// 1. Converts the fields of the dataset from string to the field types of the covtype_rec (below).
//-------------------------------------------------------------------------------------------------
covtype_string_rec := RECORD
    STRING  Elevation;
    STRING  Aspect;
    STRING  Slope;
    STRING  HDistance2Hydrology;
    STRING  VDistance2Hydrology;
    STRING  HDistance2Roadways;
    STRING  Hillshade_9am;
    STRING  Hillshade_Noon;
    STRING  Hillshade_3pm;
    STRING  HDistance2FirePoints;
    STRING WildernessArea1;
    STRING WildernessArea2;
    STRING WildernessArea3;
    STRING WildernessArea4;
    STRING SoilType01;
    STRING SoilType02;
    STRING SoilType03;
    STRING SoilType04;
    STRING SoilType05;
    STRING SoilType06;
    STRING SoilType07;
    STRING SoilType08;
    STRING SoilType09;
    STRING SoilType10;
    STRING SoilType11;
    STRING SoilType12;
    STRING SoilType13;
    STRING SoilType14;
    STRING SoilType15;
    STRING SoilType16;
    STRING SoilType17;
    STRING SoilType18;
    STRING SoilType19;
    STRING SoilType20;
    STRING SoilType21;
    STRING SoilType22;
    STRING SoilType23;
    STRING SoilType24;
    STRING SoilType25;
    STRING SoilType26;
    STRING SoilType27;
    STRING SoilType28;
    STRING SoilType29;
    STRING SoilType30;
    STRING SoilType31;
    STRING SoilType32;
    STRING SoilType33;
    STRING SoilType34;
    STRING SoilType35;
    STRING SoilType36;
    STRING SoilType37;
    STRING SoilType38;
    STRING SoilType39;
    STRING SoilType40;
    STRING  CoverType;
END;
covtype_rec := RECORD
    unsigned  rid:=1;
    unsigned  Elevation;
    unsigned  Aspect;
    unsigned  Slope;
    unsigned  HDistance2Hydrology;
    unsigned  VDistance2Hydrology;
    unsigned  HDistance2Roadways;
    unsigned  Hillshade_9am;
    unsigned  Hillshade_Noon;
    unsigned  Hillshade_3pm;
    unsigned  HDistance2FirePoints;
    unsigned1 WildernessArea1;
    unsigned1 WildernessArea2;
    unsigned1 WildernessArea3;
    unsigned1 WildernessArea4;
    unsigned1 SoilType01;
    unsigned1 SoilType02;
    unsigned1 SoilType03;
    unsigned1 SoilType04;
    unsigned1 SoilType05;
    unsigned1 SoilType06;
    unsigned1 SoilType07;
    unsigned1 SoilType08;
    unsigned1 SoilType09;
    unsigned1 SoilType10;
    unsigned1 SoilType11;
    unsigned1 SoilType12;
    unsigned1 SoilType13;
    unsigned1 SoilType14;
    unsigned1 SoilType15;
    unsigned1 SoilType16;
    unsigned1 SoilType17;
    unsigned1 SoilType18;
    unsigned1 SoilType19;
    unsigned1 SoilType20;
    unsigned1 SoilType21;
    unsigned1 SoilType22;
    unsigned1 SoilType23;
    unsigned1 SoilType24;
    unsigned1 SoilType25;
    unsigned1 SoilType26;
    unsigned1 SoilType27;
    unsigned1 SoilType28;
    unsigned1 SoilType29;
    unsigned1 SoilType30;
    unsigned1 SoilType31;
    unsigned1 SoilType32;
    unsigned1 SoilType33;
    unsigned1 SoilType34;
    unsigned1 SoilType35;
    unsigned1 SoilType36;
    unsigned1 SoilType37;
    unsigned1 SoilType38;
    unsigned1 SoilType39;
    unsigned1 SoilType40;
    unsigned  CoverType;
END;

in_ds:=DATASET('~thumphrey::covtype.data', covtype_string_rec,CSV(separator(',')));
//OUTPUT(COUNT(in_ds),NAMED('size_of_in_ds'));

d0 :=
   PROJECT(in_ds  
           ,TRANSFORM(covtype_rec
                      ,SELF.rid :=	COUNTER
                      ,SELF.Elevation := (integer)LEFT.Elevation
                      ,SELF.Aspect := (integer)LEFT.Aspect
                      ,SELF.Slope := (integer)LEFT.Slope
                      ,SELF.HDistance2Hydrology := (integer)LEFT.HDistance2Hydrology
                      ,SELF.VDistance2Hydrology := (integer)LEFT.VDistance2Hydrology
                      ,SELF.HDistance2Roadways := (integer)LEFT.HDistance2Roadways
                      ,SELF.Hillshade_9am := (integer)LEFT.Hillshade_9am
                      ,SELF.Hillshade_Noon := (integer)LEFT.Hillshade_Noon
                      ,SELF.Hillshade_3pm := (integer)LEFT.Hillshade_3pm
                      ,SELF.HDistance2FirePoints := (integer)LEFT.HDistance2FirePoints
                      ,SELF.WildernessArea1 := (integer1)LEFT.WildernessArea1
                      ,SELF.WildernessArea2 := (integer1)LEFT.WildernessArea2
                      ,SELF.WildernessArea3 := (integer1)LEFT.WildernessArea3
                      ,SELF.WildernessArea4 := (integer1)LEFT.WildernessArea4
                      ,SELF.SoilType01 := (integer1)LEFT.SoilType01
                      ,SELF.SoilType02 := (integer1)LEFT.SoilType02
                      ,SELF.SoilType03 := (integer1)LEFT.SoilType03
                      ,SELF.SoilType04 := (integer1)LEFT.SoilType04
                      ,SELF.SoilType05 := (integer1)LEFT.SoilType05
                      ,SELF.SoilType06 := (integer1)LEFT.SoilType06
                      ,SELF.SoilType07 := (integer1)LEFT.SoilType07
                      ,SELF.SoilType08 := (integer1)LEFT.SoilType08
                      ,SELF.SoilType09 := (integer1)LEFT.SoilType09
                      ,SELF.SoilType10 := (integer1)LEFT.SoilType10
                      ,SELF.SoilType11 := (integer1)LEFT.SoilType11
                      ,SELF.SoilType12 := (integer1)LEFT.SoilType12
                      ,SELF.SoilType13 := (integer1)LEFT.SoilType13
                      ,SELF.SoilType14 := (integer1)LEFT.SoilType14
                      ,SELF.SoilType15 := (integer1)LEFT.SoilType15
                      ,SELF.SoilType16 := (integer1)LEFT.SoilType16
                      ,SELF.SoilType17 := (integer1)LEFT.SoilType17
                      ,SELF.SoilType18 := (integer1)LEFT.SoilType18
                      ,SELF.SoilType19 := (integer1)LEFT.SoilType19
                      ,SELF.SoilType20 := (integer1)LEFT.SoilType20
                      ,SELF.SoilType21 := (integer1)LEFT.SoilType21
                      ,SELF.SoilType22 := (integer1)LEFT.SoilType22
                      ,SELF.SoilType23 := (integer1)LEFT.SoilType23
                      ,SELF.SoilType24 := (integer1)LEFT.SoilType24
                      ,SELF.SoilType25 := (integer1)LEFT.SoilType25
                      ,SELF.SoilType26 := (integer1)LEFT.SoilType26
                      ,SELF.SoilType27 := (integer1)LEFT.SoilType27
                      ,SELF.SoilType28 := (integer1)LEFT.SoilType28
                      ,SELF.SoilType29 := (integer1)LEFT.SoilType29
                      ,SELF.SoilType30 := (integer1)LEFT.SoilType30
                      ,SELF.SoilType31 := (integer1)LEFT.SoilType31
                      ,SELF.SoilType32 := (integer1)LEFT.SoilType32
                      ,SELF.SoilType33 := (integer1)LEFT.SoilType33
                      ,SELF.SoilType34 := (integer1)LEFT.SoilType34
                      ,SELF.SoilType35 := (integer1)LEFT.SoilType35
                      ,SELF.SoilType36 := (integer1)LEFT.SoilType36
                      ,SELF.SoilType37 := (integer1)LEFT.SoilType37
                      ,SELF.SoilType38 := (integer1)LEFT.SoilType38
                      ,SELF.SoilType39 := (integer1)LEFT.SoilType39
                      ,SELF.SoilType40 := (integer1)LEFT.SoilType40
                      ,SELF.CoverType := (integer)LEFT.CoverType
            )
   );

//-------------------------------------------------------------------------------------------------
// 2. Makes the dataset suitable for Logistic make sure the dependent variable/field, CoverType has only 2 values: 0 or 1.
//-------------------------------------------------------------------------------------------------

// Make dependent variable either 1 (if CoverType=2) or 0 otherwise. Did this so Logistic could use it during learning.
ForestCoverType:=PROJECT(d0,TRANSFORM(RECORDOF(d0),SELF.CoverType:=IF(LEFT.CoverType=2,1,0),SELF:=LEFT));

training_size := COUNT(ForestCoverType);
OUTPUT(training_size,NAMED('training_size'));

//-------------------------------------------------------------------------------------------------
// 3. Separate the independent variables/fields from the dependent variables/fields
//-------------------------------------------------------------------------------------------------

// record layout for independent features
indep_rec := RECORD
ForestCoverType;
NOT ForestCoverType.CoverType;
END;
indep_Data:= TABLE(ForestCoverType,indep_rec);
//OUTPUT(indep_Data,NAMED('indep_Data'));
dep_Data:= TABLE(ForestCoverType,{rid, CoverType});
//OUTPUT(dep_Data,NAMED('dep_Data'));

train_indep_Data := indep_Data[1 .. training_size];
train_dep_Data := dep_Data[1 .. training_size];

//-------------------------------------------------------------------------------------------------
// 4. Convert both the independent and dependent datasets to the record format used by ML, i.e. NumericField
//-------------------------------------------------------------------------------------------------

ToField(train_indep_Data, pr_train_indep);
train_indepData := ML.Discretize.ByRounding(pr_train_indep)(value<>0); //The filter, (value<>0), makes it sparse.
ToField(train_dep_Data, pr_train_dep);
train_depData := ML.Discretize.ByRounding(pr_train_dep);

//-------------------------------------------------------------------------------------------------
// 5. Do Logistic Regression learning to produce a model.
//-------------------------------------------------------------------------------------------------

learner := ML.Classify.Logistic();
result := learner.LearnD(train_indepData, train_depData); // model to use when classifying
//OUTPUT(result,NAMED('result'));
model:= learner.model(result);  // transforming model to a easier way to read it
OUTPUT(model,NAMED('model_output')); // group_id represent number of tree
