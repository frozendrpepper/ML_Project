# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:01:28 2017

@author: cck3
"""

'''Import all the necessary packages'''
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

from patsy import dmatrix
from scipy import stats

import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def graph_na(data_train):
    plt.figure()
    missing_data_count = data_train.isnull().sum()
    missing_data_count.sort_values(ascending = False, inplace = True)
    missing_data_count_filtered = missing_data_count[missing_data_count > 0]
    missing_data_count_filtered.plot(kind = 'bar', title = 'missing data count')
    plt.show()
    
def count_outcome(data_train):
    plt.figure()
    outcome = data_train['OutcomeType'].value_counts()
    outcome.plot(kind = 'bar', title = 'Outcome variable count')
    plt.show()

def no_name_label(x):
    '''Reference: https://github.com/JihongL/Shelter-Animal-Outcomes/blob/master/Shelter_EDA.ipynb'''
    if type(x) == float:
        return 0
    else:
        return 1
    
def datetime_converter(data_train):
    datetime = list(data_train['DateTime'])
    year_list, month_list = [], []
    for item in datetime:
        year_list.append(int(item[:4]))
        month_list.append(int(item[5:7]))
    return year_list, month_list

def season_sort(month_list):
    season_list = []
    for item in month_list:
        if item == 12 or item == 1 or item == 2:
            season_list.append('winter')
        elif item >=3 and item <=5:
            season_list.append('spring')
        elif item >= 6 and item <= 8:
            season_list.append('summer')
        elif item >= 9 and item <= 11:
            season_list.append('fall')
    return season_list

def datetime_outcome(data_train):
    #plt.figure()
    #datetime_groupby = pd.DataFrame(data_train.groupby(['OutcomeYear', 'OutcomeMonth', 'OutcomeType'])['OutcomeType'].size())
    pass

def convert_Age(data_train):
    '''Reference: https://stackoverflow.com/questions/12851791/removing-numbers-from-string'''
    age_list = list(data_train['AgeuponOutcome'])
    age_day_compile = []
    for item in age_list:
        if type(item) != str:
            age_day_compile.append(item)
        else:
            numeric = int(item[:2])
            if 'year' in item:
                age_day_compile.append(numeric * 365)
            elif 'month' in item:
                age_day_compile.append(numeric * 30)
            elif 'week' in item:
                age_day_compile.append(numeric * 7)
            elif 'day' in item:
                age_day_compile.append(numeric)
    return age_day_compile        
    
    
data_train = pd.read_csv('train.csv')
'''Keep a copy of the original data for comparison'''
data_original = data_train[:]

'''Find out how many NaN values are in the dataset'''
graph_na(data_train)

'''Drop the AnimalID column since we don't really need it'''
data_train.drop('AnimalID', axis = 1, inplace = True)

'''Count Number of outcomes'''
count_outcome(data_train)

'''Simple data processing for pets with name get 1 and pets without names get a 0'''
data_train['Name'] = data_train['Name'].transform(no_name_label)

'''Animal mapping'''
animal_type_mapping = {'Dog':1, 'Cat':0}
data_train['AnimalType'] = data_train['AnimalType'].map(animal_type_mapping)

'''Outcome mapping'''
outcome_mapping = {'Return_to_owner':1, 'Euthanasia':2, 'Adoption':3, 'Transfer':4, 'Died':5}
data_train['OutcomeType'] = data_train['OutcomeType'].map(outcome_mapping)

'''Year and Month information extraction'''
year_list, month_list = datetime_converter(data_train)
data_train['OutcomeYear'], data_train['OutcomeMonth'] = year_list, month_list
season_list = season_sort(month_list)
data_train['OutcomeSeason'] = season_list

'''Convert AgeUponOutcome to unit of days'''
'''First confirm the unique string values that are present in the column'''
'''Reference: https://stackoverflow.com/questions/12851791/removing-numbers-from-string'''
age_list_compile = []
age_outcome_list = list(data_train['AgeuponOutcome'])
for item in age_outcome_list:
    if type(item) != str:
        '''This condition is to account for NaN values'''
        continue
    result = ''.join(i for i in item if not i.isdigit())
    age_list_compile.append(result)
'''We can check unique string values in the column'''
age_list_unique = list(pd.Series(age_list_compile).unique())

age_day_compile = convert_Age(data_train)
data_train['AgeuponOutcome'] = age_day_compile

'''Rearrange SexUponOutcome variables'''
'''If they are sprayed/neutered, consider them equal as asexual'''