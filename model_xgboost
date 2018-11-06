import pandas as pd
import numpy as np
import datetime
from pandas import DataFrame
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor   # this is for regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from scipy.stats import ranksums
from sklearn.metrics import precision_recall_fscore_support
import pickle


def model(idx,train_data):
    r_auc = []
    r_acc = []
    r_pre = []
    r_rec = []
    for t in range(10):  # cross validation
        X, X_hold, Y, Y_hold = train_test_split(train_data, idx, test_size=0.5, random_state=t)
        pos_ind = np.where(Y == 1)[0]
        neg_ind = np.where(Y == 0)[0]
        pval = [ranksums(X[pos_ind, x],X[neg_ind, x]).pvalue for x in range(X.shape[1])]
        sorted_pval = sorted(pval)
        fe_num = min(99, len(sorted_pval)-1)
        index = np.where(pval <= sorted_pval[fe_num])[0]
    #     print(index)

        X = X[:,index]
        X_hold = X_hold[:,index]
        model = XGBClassifier(learning_rate =0.1, n_estimators=100, seed=t)
        model.fit(X, Y)
        importances = model.feature_importances_
        while any(importances == 0):
            X = X[:,importances>0]
            X_hold = X_hold[:,importances>0]
            model.set_params(n_estimators=np.sum(importances>0))
            model.fit(X, Y)
            importances = model.feature_importances_
        y_pred = model.predict_proba(X_hold)[:,1]
        predictions = [round(value) for value in y_pred]
        auc = roc_auc_score(Y_hold,y_pred)
        accuracy = accuracy_score(Y_hold, predictions)
        tmp = precision_recall_fscore_support(Y_hold,predictions)
        r_auc.append(auc)
        r_acc.append(accuracy)
        r_pre.append(tmp[0][1])
        r_rec.append(tmp[1][1])
        print(t, auc, accuracy, tmp[0][1], tmp[1][1])
    #Accuracy=np.mean(r_acc) * 100.0
    #AUC= np.mean(r_auc)
    #Precision=np.mean(r_pre)
    #Recall=np.mean(r_rec)
    #result=[Accuracy,AUC,Precision,Recall]
    print("Random Split Accuracy: %.2f%%" % (np.mean(r_acc) * 100.0))
    #print("Random Split Accuracy std: %.2f" % (np.std(r_acc) ))
    print("Random Split AUC: %.2f" % ( np.mean(r_auc) ))
    #print("Random Split AUC std: %.2f" % ( np.std(r_auc) ))
    print("Random Split Precision: %.2f" % ( np.mean(r_pre) ))
    print("Random Split Recall: %.2f" % ( np.mean(r_rec) ))
    
    # model building
    pos_ind = np.where(idx == 1)[0]
    neg_ind = np.where(idx == 0)[0]
    pval = [ranksums(train_data[pos_ind, x],train_data[neg_ind, x]).pvalue for x in range(train_data.shape[1])]
    sorted_pval = sorted(pval)
    fe_num = min(99, X.shape[1])
    
    index = np.where(pval <= sorted_pval[fe_num])[0]
    train_data = train_data[:,index]
    select_marker = np.array(range(train_data.shape[1]))
    model = XGBClassifier(learning_rate =0.1, n_estimators=100, seed=7)
    model.fit(train_data, idx )
    importances = model.feature_importances_
    while any(importances == 0):
        train_data = train_data[:,importances>0]
        select_marker = select_marker[importances>0]
        model.set_params(n_estimators=np.sum(importances>0))
        model.fit(train_data,idx)
        importances = model.feature_importances_ 
    model_path = './xgboost_10.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return importances, select_marker, index, train_data
   #return importances, select_marker, index, train_data,result
