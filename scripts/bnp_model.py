#!/usr/bin/python
###################################################################################################################
### This code is developed by HighEnergyDataScientests Team.
### Do not copy or modify without written approval from one of the team members.
###################################################################################################################

import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt
import time

#seed = 260681

### Controlling Parameters
output_col_name = "target"
test_col_name = "PredictedProb"
enable_feature_analysis = 1
id_col_name = "ID"


def ceate_feature_map(features,featureMapFile):
    outfile = open(featureMapFile, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

print("## Loading Data")
train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')


print("## Data Processing")
train = train.drop(id_col_name, axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

print("## Data Encoding")
for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

features = [s for s in train.columns.ravel().tolist() if s != output_col_name]
print("Features: ", features)


print("## Training")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "nthread":-1,
          "max_depth": 10,
          "subsample": 0.8,
          "colsample_bytree": 0.8,
          "eval_metric": "logloss",
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 500

print("Train a XGBoost model")
timestr = time.strftime("%Y%m%d-%H%M%S")
X_train, X_valid = train_test_split(train, test_size=0.1)
y_train = X_train[output_col_name]
y_valid = X_valid[output_col_name]
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, maximize=True, early_stopping_rounds=100, verbose_eval=True)


if enable_feature_analysis == 1:
    print("## Creating Feature Importance Map")
    featureMapFile = '../feature_analysis/xgb_' + timestr +'.fmap'
    ceate_feature_map(features,featureMapFile)
    importance = gbm.get_fscore(fmap=featureMapFile)
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 12))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    fig_featp = featp.get_figure()
    fig_featp.savefig('../feature_analysis/feature_importance_xgb_' + timestr + '.png',bbox_inches='tight',pad_inches=1)
    df.to_csv('../feature_analysis/feature_importance_xgb_' + timestr + '.csv')

print("## Predicting test data")
preds = gbm.predict(xgb.DMatrix(test[features]),ntree_limit=gbm.best_ntree_limit)
test[test_col_name] = preds
test[[id_col_name,test_col_name]].to_csv("../predictions/pred_" + timestr + ".csv", index=False)
