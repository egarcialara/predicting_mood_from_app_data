#!/usr/bin/python

"""
Author: Elena Garcia
Course: Data Mining Techniques


Support Vector Regression

"""


from __future__ import division
import sklearn
from sklearn.model_selection import KFold, cross_val_score, permutation_test_score, StratifiedKFold, GridSearchCV, cross_val_predict, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from itertools import cycle
from sklearn.svm import SVR
import numpy as np
import itertools
import random
import math
import sys
from inspect_data import main as main_inspect
import pandas

# Tuning parameters of SVM
def tuning_parameters(X_train, y_train):

    # Set the parameters by cross-validation
    svc = SVR()
    c_values  = [0.1, 5, 10 ,15,10,25, 50, 100, 200,300,400,500,600,700,1000]
    gamma_values = [1e-1,1e-2, 1e-3, 1e-4, 1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
    parameters = {'kernel':('linear', 'poly','rbf', 'sigmoid'), 'C': c_values, 'gamma':gamma_values }

    clf = GridSearchCV(svc, param_grid=parameters, cv=10, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)

    print "Grid scores on development set: "

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    	print "%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params)

    print "Best parameters set found on training set:"
    print clf.best_params_
    print "\n"

    # # Based on precison:
    # Best Parameters: {'kernel': 'rbf', 'C': 10, 'gamma': 0.001, 'class_weight': None}
    # Based on recall:
    # Best Parameters: {'kernel': 'rbf', 'C': 100, 'gamma': 0.001, 'class_weight': None}

    # # Cross-Validation plot
    # CVpred = cross_val_predict(clf,  X_train, y_train, cv=10)
    # CVplot(pred_lb, CVpred, CVscores)

    return clf.best_params_

# Training our model
def SVM(X_train, Y_train, X_test, Y_test,parameters):

    # k-fold CV / choosing k
    folds = 10
    # Calling the SMV algorithm
    svr_rbf = SVR(kernel=parameters['kernel'], C=parameters['C'], gamma=parameters['gamma'])
    # Using K-fold cross validation step
    k_fold = KFold(n_splits=folds, random_state=True)
    svr_rbf.fit(X_train, Y_train)
    # Fitting the data to the algorithm
    #[clf.fit(X_train[train], Y_train[train]).score(X_train[test], Y_train[test]) for train, test in k_fold.split(X_train)]

    #title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    # plot_learning_curve(clf, title, X_train, Y_train, (0.7, 1.01), cv=folds, n_jobs=-1)
    # plt.show()

    # Get the predicted labels and scores
    predicted_labels = svr_rbf.predict(X_test)
    scores = svr_rbf.score(X_test,Y_test)

    # Obtaining the distances from the hyperplance
    #distance = svr_rbf.decision_function(X_test)
    pred_lb = predicted_labels.tolist()
    RMSE = mean_absolute_error(Y_test,predicted_labels)
    return pred_lb, RMSE

# Creates a test set and a training set, and returns the variables values
# for the test and training set as well as the labels obtained by the benchmarking
# for both sets
def create_testset(individual,dataframe):
    df = dataframe[individual]
    print df
    print 'this is individual:'
    print individual
    indices_list = df.index.tolist()
    # Random sampling from the whole dataset, for building the test set
    rsample = random.sample(range(len(indices_list)),int(len(indices_list)*0.3))
    testset = df.iloc[rsample]
    df.drop(df.index[rsample], inplace=True)

    Y_train = df['targetmood'].tolist()
    Y_benchmark_train = df['benchmark'].tolist()
    del df['targetmood']
    del df['benchmark']
    df_norm = (df - df.mean()) / (df.max() - df.min())
    df_norm = df_norm.fillna(0)
    X_train = df_norm.as_matrix()

    Y_test = testset['targetmood'].tolist()
    Y_test_benchmark = testset['benchmark'].tolist()
    del testset['targetmood']
    del testset['benchmark']
    testset = (testset - testset.mean()) / (testset.max() - testset.min())
    testset = testset.fillna(0)
    X_test = testset.as_matrix()

    return X_train, Y_train, Y_benchmark_train,Y_test_benchmark, X_test, Y_test


def main():

    ndframe, ndframe_withoutNA, ndframe_notimewindow, ndframe_notimewindow_withoutNA, timewindow, individuals_list = main_inspect()

    # Create global dataframe for creating a global model for all individuals
    first = True
    for individual in ndframe_withoutNA.keys():
        if first:
            globaldf = ndframe_withoutNA[individual]
        else:
            globaldf = pandas.concat([globaldf,ndframe_withoutNA[individual]])


    # X_train, Y_train, Y_benchmark_train,Y_test_benchmark, X_test, Y_test = create_testset('AS14.01',ndframe_withoutNA)
    #
    # # Tuning parameters of SVM
    # print 'tuning parameters of SVM'
    # best_parameters = tuning_parameters(X_train, Y_train)
    # print "Best Parameters: %s" %str(best_parameters)
    #
    #
    # print 'testing SVM'
    # # # Training the Support Vector Machine algorithm and get the prediction of the test set
    # predicted_labels = SVM(X_train, Y_train, X_test, Y_test,best_parameters)
    # print 'benchmark'
    # print mean_absolute_error(Y_test,Y_test_benchmark)

    f = open('results/SVR_twindow_%s.dat'%(str(timewindow)),'w')
    f.write('#Individual\tRMSE_SVR\tRMSE_Benchmark\tSVR_parameters\ttimewindow=%s\n'%(str(timewindow)))

    # for individual in individuals_list:
    #     X_train, Y_train, Y_benchmark_train,Y_test_benchmark, X_test, Y_test = create_testset(individual,ndframe_withoutNA)

    # # Tuning parameters of SVM
    #     print 'tuning parameters of SVM for individual=%s'%(str(individual))
    #     best_parameters = tuning_parameters(X_train, Y_train)
    #     print "Best Parameters: %s" %str(best_parameters)


    # # Training the Support Vector Machine algorithm and get the prediction of the test set
    #     predicted_labels, RMSE = SVM(X_train, Y_train, X_test, Y_test,best_parameters)
    #     print 'benchmark'
    #     benchmark_error = mean_absolute_error(Y_test,Y_test_benchmark)
    #     f.write('%s\t%s\t%s\t%s\n'%(str(individual),str(RMSE),str(benchmark_error),str(best_parameters)))


    # Build a global model and print the results in the file
    X_train, Y_train, Y_benchmark_train,Y_test_benchmark, X_test, Y_test = create_testset(individual,ndframe_withoutNA)
    # Tuning parameters of SVM
    print 'tuning parameters of SVM for individual=%s'%s(str(individual))
    best_parameters = tuning_parameters(X_train, Y_train)
    print "Best Parameters: %s" %str(best_parameters)

    # Training the Support Vector Machine algorithm and get the prediction of the test set
    predicted_labels, RMSE = SVM(X_train, Y_train, X_test, Y_test,best_parameters)
    print 'benchmark'
    benchmark_error = mean_absolute_error(Y_test,Y_test_benchmark)
    f.write('%s\t%s\t%s\t%s\n'%(str(individual),str(RMSE),str(benchmark_error),str(best_parameters)))
    f.close()



if __name__ == "__main__":
    main()
