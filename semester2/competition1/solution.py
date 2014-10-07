import pandas
import sys
import numpy as np
import scipy.sparse
# append the path to xgboost, you may need to change the following line
# alternatively, you can add the path to PYTHONPATH environment variable
sys.path.append('/home/yuki/xgboost-master/wrapper')
import xgboost as xgb


train = pandas.read_csv('molecules_train.csv')
print 'Data size:', train.shape
train.ix[:, :16].head()

test = pandas.read_csv('molecules_test.csv')
print 'Data size:', test.shape
test.ix[:, :16].head()


y = train.ix[:, 0].values
X = train.ix[:, 1:].values
Xt = test.ix[:, 1:].values


def save_predictions(predictions):
    pandas.Series(data=predictions, name='PredictedProbability', index=test.MoleculeId).to_csv('test_output.csv', header=True)    






dtrain = xgb.DMatrix( X, label=y)
dtest = xgb.DMatrix(Xt)
param = {'max_depth':10, 'eta':0.1, 'silent':1, 'subsample':0.5,'objective':'binary:logistic', 'eval_metric':'logloss'}
num_round = 12


xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'logloss'}, seed=0, show_stdv = False)


bst = xgb.train(param, dtrain, 10)
preds = bst.predict(dtest)

if len(preds.shape) > 1:
        preds = preds[:, 1]
save_predictions(preds)