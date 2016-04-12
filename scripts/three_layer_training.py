#!/usr/bin/python
###################################################################################################################
### This code is developed by DataTistics team on kaggle
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################
#
############################### This is three_layer_training module ###############################################

import numpy as  np

from sklearn.metrics import log_loss

from time import time

#from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.calibration import CalibratedClassifierCV
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
#from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

def LgR_Classifier (X_train, X_valid, y_train, y_valid):
    print("## Fitting the classifier to the training set")
    t0 = time()
    clf = LogisticRegressionCV(scoring='log_loss')
    clf = clf.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    
    # make predictions
    print("## Making Predictions")
    p_train = clf.predict_proba(X_train)
    p_valid = clf.predict_proba(X_valid)
    
    score_train = log_loss(y_train, p_train[:,1])
    score_valid = log_loss(y_valid, p_valid[:,1])
    print("Score based on training data set = ", score_train)
    print("Score based on validating data set = ", score_valid)

#fixing random state
random_state=1

def First_layer_classifiers (X, X_train, X_valid, X_test, y, y_train, y_valid, y_test, temp_test):
    #Defining the classifiers
    clfs = {'LRC'  : LogisticRegression(n_jobs=-1, random_state=random_state), 
            'SVM' : SVC(probability=True, max_iter=100, random_state=random_state), 
            'RFC'  : RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                       random_state=random_state), 
            'GBM' : GradientBoostingClassifier(n_estimators=50, 
                                           random_state=random_state), 
            'ETC' : ExtraTreesClassifier(n_estimators=100, n_jobs=-1, 
                                     random_state=random_state),
            'KNN' : KNeighborsClassifier(n_neighbors=30, n_jobs=-1),
            'ABC' : AdaBoostClassifier(DecisionTreeClassifier(max_depth=30),
                                       algorithm="SAMME", n_estimators=350,
                                       random_state=random_state),
            'SGD' : SGDClassifier(loss="log", n_iter = 1100, n_jobs=-1,
                                  random_state=random_state),
            'DTC' : DecisionTreeClassifier(max_depth=7, random_state=random_state)
            }
            
    
    #predictions on the validation and test sets
    p_valid = []
    p_test = []
    p_ttest = []
    p_ttest_t = []
    
    print('')
    print('Performance of individual classifiers (1st layer) on X_test')   
    print('-----------------------------------------------------------')
   
    for nm, clf in clfs.items():
       #First run. Training on (X_train, y_train) and predicting on X_valid.
       clf.fit(X_train, y_train)
       yv = clf.predict_proba(X_valid)
       p_valid.append(yv)
       
       yv_tt = clf.predict_proba(temp_test)
       p_ttest.append(yv_tt)
       
       #Printing out the performance of the classifier
       print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(nm), 'logloss_val  =>', log_loss(y_valid, yv[:,1])))
       
       #Second run. Training on (X, y) and predicting on X_test.
       clf.fit(X, y)
       yt = clf.predict_proba(X_test)
       p_test.append(yt)
       
       yt_tt = clf.predict_proba(temp_test)
       p_ttest_t.append(yt_tt)
       
       #Printing out the performance of the classifier
       print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(nm), 'logloss_test  =>', log_loss(y_test, yt[:,1])))
       print('')

    return p_valid, p_test, p_ttest, p_ttest_t


def Second_layer_ensembling (p_valid, p_test, y_valid, y_test, p_ttest, p_ttest_t):
    print('')
    print('Performance of optimization based ensemblers (2nd layer) on X_test')
    print('------------------------------------------------------------------')
    
    #Creating the data for the 2nd layer
    XV = np.hstack(p_valid)
    XT = np.hstack(p_test)
    
    XTT = np.hstack(p_ttest)
    XTTT = np.hstack(p_ttest_t)
    
    clf = LogisticRegressionCV(scoring='log_loss', random_state=random_state)
    clf = clf.fit(XV, y_valid)
    
    yT = clf.predict_proba(XT)
    y_out = clf.predict_proba(XTT)
    yt_out = clf.predict_proba(XTTT)
    print('{:20s} {:2s} {:1.7f}'.format('Ensemble of Classifiers', 'logloss_ensembled  =>', log_loss(y_test, yT[:,1])))
    
    return y_out, yt_out
