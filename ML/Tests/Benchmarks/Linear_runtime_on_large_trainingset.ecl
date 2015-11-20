IMPORT ML;
/*
EXECUTION TIME OF THIS CODE: It took 3 minutes and 9 seconds on a 20 node THOR
to train (create model) for a training set having 100 million observations. 
Here are the specs of the THOR:

CPU = Intel(R) Xeon(TM) CPU 3.20GHz
RAM = 4GB DDR2
NIC = 1Gbit

*/

TestSize := 100000000;

a1 := ML.Distribution.Uniform(0,100,10000); 
X1 := ML.Distribution.GenData(TestSize,a1,1); // Field 1 Uniform
// Field 2 Normally Distributed
a2 := ML.Distribution.Normal2(0,10,10000);
X2 := ML.Distribution.GenData(TestSize,a2,2);
// Field 3 - Poisson Distribution
a3 := ML.Distribution.Poisson(4,100);
Y0 := ML.Distribution.GenData(TestSize,a3,3);

D := X1+X2+Y0; // This is the test data
OUTPUT(COUNT(D),NAMED('Size_of_D'));
OUTPUT(SORT(D,id,number),NAMED('D'));

X := D(Number <= 2); // Pull out the X
OUTPUT(COUNT(X),NAMED('size_of_X'));
Y := D(Number = 3); // Pull out the Y
model_dense := ML.Regression.dense.OLS_LU(X,Y);
OUTPUT(model_dense.Betas,NAMED('DenseModel'));
