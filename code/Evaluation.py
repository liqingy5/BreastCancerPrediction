import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
import pickle
import metrics


def evaluate(model_name):
    import utils
    x_train,x_test,y_train,y_test=utils.load_standard_data()
    # Load the Model back from file
    with open(f'../models/{model_name}.model', 'rb') as file:  
        model = pickle.load(file)
        
#     if model_name =="XGboost":
#         dtest = xgb.DMatrix(data=X_test, label=y_test)
#         y_pred_probs = model.predict(dtest)
#         metrics.roc_pr_curve(y_test,y_pred_probs)
#         y_pred_probs[y_pred_probs >= 0.5] = 1
#         y_pred_probs[y_pred_probs < 0.5] = 0
#         metrics.conf_matrix(y_test,y_pred_probs)
#         return

    
    sc_train = model.score(x_train, y_train)
    sc_test = model.score(x_test, y_test)
    y_pred_test = model.predict(x_test)
    probs=model.predict_proba(x_test)
    print(sc_train)
    print(sc_test)

    metrics.conf_matrix(y_test,y_pred_test)
    metrics.roc_pr_curve(y_test,probs[:,1])
    



