# MachineLearning Summary
This project is a requirement for the "Practical Machine Learning" course within Coursera Data Science Specialization.

The goal of the project is to predict, given a set of measurements taken by sensors embedded in a mobile device, whether the manner in which a subject is lifting a barbell is correct or exhibits characteristics of one of four common mistakes.  That is, we are to build a classification model and use it to predict for a set of subjects for which the class is unknown.  We were asked to document our model-building strategy as well as provide predictions for a test dataset.


# Description of Files
This repository contains several files related to the analysis.  These files are briefly described below.

* __AnalysisSummary.md__: A description of the modeling process and the results obtained.
* __plmtesting.csv__: A provided dataset containing the feature vectors for 20 subjects.  The classification of each subject is unknown, and we are asked to provide predictions for these subjects.
* __plmtraining.csv__: The provided dataset for training the model. This should not be confused with the data actually used to train the model. This is simply a dataset provided for which the classification of each subject is known.
* __run_analysis.R__: Loads the necessary data, constructs the predictive model, and provides predictions after assessing the model.

The data was provided by the course instructors, and it was originally obtained from Groupware@LES ([http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)).  The experiment, corresponding data, and subsequent publications are provided under the _Weight Lifting Exercises Data_ section.
