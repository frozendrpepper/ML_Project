# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:01:28 2017

@author: cck3
"""

'''Import all the necessary packages'''
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
#from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
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
        return 'No Name'
    else:
        return 'Name'
    
def no_name_numeric_label(x):
    '''Reference: https://github.com/JihongL/Shelter-Animal-Outcomes/blob/master/Shelter_EDA.ipynb'''
    if type(x) == float:
        return 0
    else:
        return 1
    
def convert_subtype(data_train):
    subtype_unique = list(data_train['OutcomeSubtype'].unique())
    subtype_mapping = {}
    for i, item in enumerate(subtype_unique):
        if item == 0:
            continue
        else:
            subtype_mapping[item] = i
    data_train['OutcomeSubtype'] = data_train['OutcomeSubtype'].map(subtype_mapping)
    return data_train, subtype_mapping

def datetime_str_converter(data_train):
    datetime = list(data_train['DateTime'])
    year_list, month_list = [], []
    for item in datetime:
        year_list.append(item[:4])
        month_list.append(item[5:7])
    return year_list, month_list
    
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

def convert_sex(data_train):
    sexOutcome_mapping = {'Neutered Male':1, 'Spayed Female':1, 'Intact Male':2, 'Intact Female':2, 
                          'Unknown':0} 
    data_train['SexuponOutcome'] = data_train['SexuponOutcome'].map(sexOutcome_mapping)
    return data_train, sexOutcome_mapping
    
    
data_train = pd.read_csv('train.csv')
'''Keep a copy of the original data for comparison'''
data_original = data_train.copy()

'''Create an intermediate version of data_train for easier countplot analysis'''
data_inter = data_train.copy()
data_inter['Name'] = data_inter['Name'].transform(no_name_label)
data_inter['OutcomeSubtype'] = data_inter['OutcomeSubtype'].fillna('Unknown')

'''Year and Month information extraction'''
year_list, month_list = datetime_str_converter(data_inter)
date_list = []
for year, month in zip(year_list, month_list):
    date_list.append(int(year + month))
data_inter['Date'] = date_list
year_list, month_list = datetime_converter(data_inter)
data_inter['OutcomeYear'], data_inter['OutcomeMonth'] = year_list, month_list
season_list = season_sort(month_list)
data_inter['OutcomeSeason'] = season_list
    
'''Graph some useful charts using Seaborn countplot'''
'''1) Relationship b/t sexual orientation and the outcome'''
plt.figure()
sns.countplot(x = 'SexuponOutcome', hue = 'OutcomeType', data = data_inter)
'''2) Analyze whether being a dog or not affect the outcome'''
plt.figure()
sns.countplot(x = 'AnimalType', hue = 'OutcomeType', data = data_inter)
'''3) Relationship between month and the final outcome'''
'''Reference: https://stackoverflow.com/questions/42528921/how-to-prevent-overlapping-x-axis-labels-in-sns-countplot'''
plt.figure()
ax = sns.countplot(x = 'Date', hue = 'OutcomeType', data = data_inter)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

'''4)Plot by seasonal change'''
plt.figure()
sns.factorplot(x = 'OutcomeSeason', hue = 'OutcomeYear', col = 'OutcomeType', data = data_inter, kind = 'count')

'''Find out how many NaN values are in the dataset'''
graph_na(data_train)

'''Drop the AnimalID column since we don't really need it'''
data_train.drop('AnimalID', axis = 1, inplace = True)

'''Count Number of outcomes'''
count_outcome(data_train)

'''name mapping get 1 and pets without names get a 0'''
data_train['Name'] = data_train['Name'].transform(no_name_numeric_label)

'''OutcomeSubtype encoding'''
data_train, outComeSubtype_mapping = convert_subtype(data_train)
data_train['OutcomeSubtype'] = data_train['OutcomeSubtype'].fillna(0)

'''Drop all nan values in other columns'''
data_train = data_train.dropna()

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
'''Drop the datetime column since we don't need information on day and 
the exact time of adoption'''
data_train.drop('DateTime', axis = 1, inplace = True)

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
'''This list contains year(s), month(s), week(s), day(s)'''
age_day_compile = convert_Age(data_train)
data_train['AgeuponOutcome'] = age_day_compile

'''Rearrange SexUponOutcome variables'''
'''If they are sprayed/neutered, consider them equal as neutered, if not intact.
Unknowns will be left as unknowns'''
data_train, sexOutcome_mapping = convert_sex(data_train)