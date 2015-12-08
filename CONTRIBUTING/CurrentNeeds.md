Current Needs
=============

We need two ECL verification programs for each the following learning algorithms:

 - Decision Trees
 - Random Forest
 - SoftMax
 - Deep Learning
 - Neural Networks

One program verifies the correctness of the model created by the learner of one of the above learning algorithms. And, another verifies the execution time, on large training set, of the learner of one of the above learning algorithms. The details can be found in this [JIRA](https://track.hpccsystems.com/browse/ML-266).


Examples are in the [ML.Tests.Benchmarks](https://github.com/hpcc-systems/ecl-ml/ML/Tests/Benchmarks) folder, there are two examples of ECL programs that verify the correctness of the created model and two examples of ECL programs that verify the execution time of the learner on large training sets. The following are these test ECL programs:

- Linear\_verify_model.ecl
- Logistic\_verify_model.ecl
- Linear\_runtime\_on\_large\_trainingset.ecl
- Logistic\_runtime\_on\_large\_trainingset.ecl

Two documents that describe these ECL verifications programs are in the folder, [Docs](https://github.com/hpcc-systems/ecl-ml/docs). They are LinearRegressionIntroduction.htm and LogisticRegressionIntroduction.htm.

If you have never contributed to our Machine Learning (ML) Project then read [BeginnerContributorGuide.md](https://github.com/hpcc-systems/ecl-ml/CONTRIBUTING/BeginnerContributorGuide.md).

