# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:28:47 2017

@author: cck3
"""

'''Data combination test'''

'''Import all the necessary packages'''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
import xgboost

from patsy import dmatrix
from scipy import stats

import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import requests
from bs4 import BeautifulSoup



def graph_na(data_train):
    plt.figure()
    missing_data_count = data_train.isnull().sum()
    missing_data_count.sort_values(ascending = False, inplace = True)
    missing_data_count_filtered = missing_data_count[missing_data_count > 0]
    missing_data_count_filtered.plot(kind = 'bar', title = 'missing data count')
    plt.show()
    
###############################################################################
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

dfY = data_train['OutcomeType']
data_train_drop = data_train.drop('OutcomeSubtype', axis = 1)
data_train_drop.drop('OutcomeType', axis = 1, inplace = True)

data_test = data_test.rename(columns = {'ID':'AnimalID'})
data_combined = pd.concat([data_train_drop, data_test], axis = 0, ignore_index = True)

temp = data_combined.iloc[0:26729, :]
temp2 = data_combined.iloc[26729:, :]