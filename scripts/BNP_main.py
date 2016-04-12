#!/usr/bin/python
###################################################################################################################
### This code is developed by DataTistics team on kaggle
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################
#
############################### This is BNP_main script which calls to functions from modules ##################################################

import time

# Custom modules
from datapreprocessing import train, X_train, X_valid, X_test, y, y_train, y_valid, y_test, temp_test, test
from datapreprocessing import test_col_name, id_col_name
from three_layer_training import First_layer_classifiers, Second_layer_ensembling

## Fitting the classifier to the training set
#LgR_Classifier(X_train, X_valid, y_train, y_valid)

p_valid, p_test, p_ttest, p_ttest_t = First_layer_classifiers(train, X_train, X_valid, X_test, y, y_train, y_valid, y_test, temp_test)

y_out, yt_out = Second_layer_ensembling(p_valid, p_test, y_valid, y_test, p_ttest, p_ttest_t)

# Producing output of predicted probabilities and writing it into a file
timestamp = time.strftime("%Y%m%d-%H%M%S")
test[test_col_name] = y_out[:,1]
test[[id_col_name,test_col_name]].to_csv("../predictions/2layer_" + timestamp + ".csv", index=False)

test[test_col_name] = yt_out[:,1]
test[[id_col_name,test_col_name]].to_csv("../predictions/2layer_yt_out" + timestamp + ".csv", index=False)
