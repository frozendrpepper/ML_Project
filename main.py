# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:01:28 2017

@author: cck3
"""

'''Import all the necessary packages'''
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
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
    '''This function is used in conjunction with .transform() method to convert Name and No Name 
    to numerical values'''
    if type(x) == float:
        return 0
    else:
        return 1
    
def convert_subtype(data_train):
    '''Convert OutcomeSubtype into numerical values'''
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
    '''This method converts the datetime input into year and month in
    string format. The below method is identical except it returns numerical
    value. The string format is used for graphing with countplot'''
    datetime = list(data_train['DateTime'])
    year_list, month_list, day_list, hour_list = [], [], [], []
    for item in datetime:
        year_list.append(item[:4])
        month_list.append(item[5:7])
        day_list.append(item[8:10])
        hour_list.append(item[11:13])
    return year_list, month_list, day_list, hour_list
    
def datetime_converter(data_train):
    '''Convert date time into year and month'''
    datetime = list(data_train['DateTime'])
    year_list, month_list, day_list, hour_list = [], [], [], []
    for item in datetime:
        year_list.append(int(item[:4]))
        month_list.append(int(item[5:7]))
        day_list.append(int(item[8:10]))
        hour_list.append(int(item[11:13]))
    return year_list, month_list, day_list, hour_list

def season_sort(month_list):
    '''Converts month information to season information'''
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

def convert_Age(data_train):
    '''Reference: https://stackoverflow.com/questions/12851791/removing-numbers-from-string'''
    '''This function convert all the time into unit of day'''
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
    '''This function converts sexual orientation into simple neutered, intact, unknown cases'''
    sexOutcome_mapping = {'Neutered Male':1, 'Spayed Female':1, 'Intact Male':2, 'Intact Female':2, 
                          'Unknown':0} 
    data_train['SexuponOutcome'] = data_train['SexuponOutcome'].map(sexOutcome_mapping)
    return data_train, sexOutcome_mapping

def dog_breed_category(data_train):
    '''This function converts the breed into pure mix, mix or pure'''
    breed_df = data_train[['OutcomeType', 'AnimalType', 'Breed']]
    breed_df_dog = breed_df[breed_df.AnimalType == 'Dog'].reset_index()
    breed_list = list(breed_df_dog['Breed'])
    breed_compile = []
    for item in breed_list:
        item = item.lower()
        if 'mix' in item:
            breed_compile.append('pure mix')
        elif '/' in item:
            breed_compile.append('mix')
        else:
            breed_compile.append('pure')
    breed_df_dog['Breed_dog'] = breed_compile
    return breed_df_dog
    
def size_crawler(url, selector):
    '''This is one of the two functions used for crawlign cat/dog size'''
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    breed_list = soup.select(selector)
    compile_list = []
    for breed in breed_list:
        compile_list.append(breed.get_text().lower())
    return compile_list

def size_compile(url_list, selector):
    '''This is second functions used for crawlign cat/dog size'''
    list_compile = []
    for url in url_list:
        crawler_list = size_crawler(url, selector)
        list_compile += crawler_list
    return list_compile

def remove_mix(data_train):
    '''This function removes the keyword mix from breed list. The result from
    this method is used when converting breed into different categories'''
    breed_list = list(data_train['Breed'])
    breed_compile = []
    for item in breed_list:
        item = item.lower()
        if 'mix' in item:
            item = item[:-4]
        breed_compile.append(item)
    return breed_compile

def convert_breed(type_list, breed_list, small_list, medium_list, large_list, giant_list,
                  small_cat_list, medium_cat_list, large_cat_list, domestic_cat_list):
    '''Reference: https://stackoverflow.com/questions/16380326/check-if-substring-is-in-a-list-of-strings
    that is a really smart and succinct solution for checking substring existence in a list of strings'''
    
    '''The incoming list a list of breeds with the keyword "mix" removed and converted
    to lower case. No need for extra preprocessing. Also the dog lists are all converted
    to lower case as well. Type list is to check whether the breed belongs to dog or cat'''
    if len(type_list) != len(breed_list):
        print('Two lists must have equal length!')
        return None
    
    '''Convert the list into a string format for easier processing'''
    small_combined = '\t'.join(small_list)
    medium_combined = '\t'.join(medium_list)
    large_combined = '\t'.join(large_list)
    giant_combined = '\t'.join(giant_list)
    
    breed_compile, excluded_list, mix_breed_list, cat_breed_list = [], [], [], []
    for animal_type, breed in zip(type_list, breed_list):
        if '/' in breed: #Mixed type categorization
            mix_breed_list.append(breed)
            breed_compile.append('mix')
        elif animal_type == 0: #Cat breed categorization
            cat_breed_list.append(breed)
            if breed in small_cat_list:
                breed_compile.append('small_cat')
            elif breed in medium_cat_list:
                breed_compile.append('medium_cat')
            elif breed in large_cat_list:
                breed_compile.append('large_cat')
            elif breed in domestic_cat_list:
                breed_compile.append('domestic')
        elif animal_type == 1: #Dog breed categorization
            if breed in small_combined:
                breed_compile.append('small')
            elif breed in medium_combined:
                breed_compile.append('medium')
            elif breed in large_combined:
                breed_compile.append('large')
            elif breed in giant_combined:
                breed_compile.append('giant')
            else:
                excluded_list.append(breed)
                breed_compile.append('unknown')
    return breed_compile, excluded_list, mix_breed_list, cat_breed_list

def breed_separator(breed_remove_mix):
    '''This function separates the breed into main and sub breed columns'''
    main_breed, sub_breed = [], []
    for breed in breed_remove_mix:
        if breed.count('/') == 0:
            main_breed.append(breed)
            sub_breed.append('NA')
        elif breed.count('/') == 1 or breed.count('/') == 2:
            temp_breed_list = breed.split('/')
            main_breed.append(temp_breed_list[0])
            sub_breed.append(temp_breed_list[1])
    return main_breed, sub_breed

def color_separator(color_list):
    '''The function separates color into main and sub colors'''
    main_color, sub_color = [], []
    for color in color_list:
        if color.count('/') == 0:
            main_color.append(color)
            sub_color.append('NA')
        elif color.count('/') == 1:
            temp_color_list = color.split('/')
            main_color.append(temp_color_list[0])
            sub_color.append(temp_color_list[1])
    return main_color, sub_color

def breed_separator_mix(breed_remove_mix):
    '''One information I found was that size of the dog matters a lot and,
    in mixed breeds, it can become difficult to predict how large a dog is
    going to be. So one strategy would be just to label mixed breeds as mix'''
    pure_mix_compile = []
    for breed in breed_remove_mix:
        if breed.count('/') == 0:
            pure_mix_compile.append(breed)
        else:
            pure_mix_compile.append('mix')
    return pure_mix_compile
            
###############################################################################
###############################################################################

data_train = pd.read_csv('train.csv')
'''Keep a copy of the original data for comparison'''
data_original = data_train.copy()

'''Create an intermediate version of data_train for easier countplot analysis'''
data_inter = data_train.copy()
data_inter['Name'] = data_inter['Name'].transform(no_name_label)
data_inter['OutcomeSubtype'] = data_inter['OutcomeSubtype'].fillna('Unknown')

'''Year and Month information extraction'''
year_list, month_list, day_list, hour_list = datetime_str_converter(data_inter)
date_list = []
for year, month in zip(year_list, month_list):
    date_list.append(int(year + month))
data_inter['Date'] = date_list
year_list, month_list, day_list, hour_list = datetime_converter(data_inter)
data_inter['OutcomeYear'], data_inter['OutcomeMonth'] = year_list, month_list
data_inter['OutcomeDay'], data_inter['OutcomeHour'] = day_list, hour_list
season_list = season_sort(month_list)
data_inter['OutcomeSeason'] = season_list
    
'''Classify the Outcome via larger category of dog breed'''
breed_df_dog = dog_breed_category(data_inter)

###############################################################################
###############################################################################

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

'''Same result as above graph except they are separated by the different outcome'''
ax2 = sns.factorplot(x = 'Date', col = 'OutcomeType', data = data_inter, kind = 'count')
ax2.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

'''4)Plot by seasonal change'''
sns.factorplot(x = 'OutcomeSeason', hue = 'OutcomeYear', col = 'OutcomeType', data = data_inter, kind = 'count')

'''5)Name and No Name difference'''
plt.figure()
sns.countplot(x = 'Name', hue = 'OutcomeType', data = data_inter)

'''6)Suboutcome effect on the outcome'''
plt.figure()
sns.countplot(x = 'OutcomeSubtype', hue = 'OutcomeType', data = data_inter)

'''Outcome via larger breed cateogyr'''
plt.figure()
sns.countplot(x = 'Breed_dog', hue = 'OutcomeType', data = breed_df_dog)

###############################################################################
###############################################################################
'''This section is for crawling the dog size list'''

'''Extract list of dogs for different sizes'''
small_url_list = ['http://www.dogbreedslist.info/small-dog-breeds/list_2_1.html#.Wfx1Mmi0NPY', 
                'http://www.dogbreedslist.info/small-dog-breeds/list_2_2.html#.Wfx1Mmi0NPY', 
                'http://www.dogbreedslist.info/small-dog-breeds/list_2_3.html#.Wfx1Mmi0NPY', 
                'http://www.dogbreedslist.info/small-dog-breeds/list_2_4.html#.Wfx1Mmi0NPY', 
                'http://www.dogbreedslist.info/small-dog-breeds/list_2_5.html#.Wfx1Mmi0NPY',
                'http://www.dogbreedslist.info/small-dog-breeds/list_2_6.html#.Wfx1Mmi0NPY']

selector = 'body > div.main > div.main-r > div > div.list-01 > div.right > div.right-t > p > a'
small_dog_compile = size_compile(small_url_list, selector)

medium_url_list = ['http://www.dogbreedslist.info/medium-dog-breeds/list_3_1.html#.Wfx3bmi0NPY',
                   'http://www.dogbreedslist.info/medium-dog-breeds/list_3_2.html#.Wfx3bmi0NPY',
                   'http://www.dogbreedslist.info/medium-dog-breeds/list_3_3.html#.Wfx3bmi0NPY', 
                   'http://www.dogbreedslist.info/medium-dog-breeds/list_3_4.html#.Wfx3bmi0NPY',
                   'http://www.dogbreedslist.info/medium-dog-breeds/list_3_5.html#.Wfx3bmi0NPY',
                   'http://www.dogbreedslist.info/medium-dog-breeds/list_3_6.html#.Wfx3bmi0NPY']

selector = 'body > div.main > div.main-r > div > div.list-01 > div.right > div.right-t > p > a'
medium_dog_compile = size_compile(medium_url_list, selector) + ['treeing cur', 'treeing tennesse brindle']

large_url_list = ['http://www.dogbreedslist.info/large-dog-breeds/list_4_1.html#.Wfx9nWi0NPY',
                   'http://www.dogbreedslist.info/large-dog-breeds/list_4_2.html#.Wfx9nWi0NPY',
                   'http://www.dogbreedslist.info/large-dog-breeds/list_4_3.html#.Wfx9nWi0NPY', 
                   'http://www.dogbreedslist.info/large-dog-breeds/list_4_4.html#.Wfx9nWi0NPY',
                   'http://www.dogbreedslist.info/large-dog-breeds/list_4_5.html#.Wfx9nWi0NPY',
                   'http://www.dogbreedslist.info/large-dog-breeds/list_4_6.html#.Wfx9nWi0NPY']

selector = 'body > div.main > div.main-r > div > div.list-01 > div.right > div.right-t > p > a'
large_dog_compile = size_compile(large_url_list, selector) + ['schnauzer giant', 'olde english bulldogge']

giant_url_list = ['http://www.dogbreedslist.info/giant-dog-breeds/list_5_1.html#.Wfx9nWi0NPY',]

selector = 'body > div.main > div.main-r > div > div.list-01 > div.right > div.right-t > p > a'
giant_dog_compile = size_compile(giant_url_list, selector)
###############################################################################
###############################################################################

'''Find out how many NaN values are in the dataset'''
graph_na(data_train)

'''Drop the AnimalID column since we don't really need it'''
data_train.drop('AnimalID', axis = 1, inplace = True)

'''Count Number of outcomes'''
count_outcome(data_train)

'''name mapping get 1 and pets without names get a 0'''
data_train['Name'] = data_train['Name'].transform(no_name_numeric_label)

'''OutcomeSubtype encoding. We do not need this part so I will comment it out.
But this causes issues later when dropping nan values'''
data_train, outComeSubtype_mapping = convert_subtype(data_train)
data_train['OutcomeSubtype'] = data_train['OutcomeSubtype'].fillna(0)

'''Drop all nan values in other columns'''
data_train = data_train.dropna()
data_inter = data_inter.dropna()

'''Animal mapping'''
animal_type_mapping = {'Dog':1, 'Cat':0}
data_train['AnimalType'] = data_train['AnimalType'].map(animal_type_mapping)

'''Outcome mapping'''
outcome_mapping = {'Return_to_owner':1, 'Euthanasia':2, 'Adoption':3, 'Transfer':4, 'Died':5}
data_train['OutcomeType'] = data_train['OutcomeType'].map(outcome_mapping)

'''Year and Month information extraction'''
'''Make columns in both string an integer format as this will be useful later on'''
year_list, month_list, day_list, hour_list = datetime_str_converter(data_train)
data_train['OutcomeYearstr'], data_train['OutcomeMonthstr'] = year_list, month_list
data_train['OutcomeDaystr'], data_train['OutcomeHourstr'] = day_list, hour_list
date_list = []
for year, month in zip(year_list, month_list):
    date_list.append(int(year + month))
data_train['Date'] = date_list

year_list, month_list, day_list, hour_list = datetime_converter(data_train)
data_train['OutcomeYear'], data_train['OutcomeMonth'] = year_list, month_list
data_train['OutcomeDay'], data_train['OutcomeHour'] = day_list, hour_list
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

'''Create a column that converts Breed into size category'''
'''First remove the string "mix" from breed category'''
breed_remove_mix = remove_mix(data_train)

'''This preprocessing was done to match some of the names that were different 
from the list obtained on the website'''
for index, item in enumerate(breed_remove_mix):
    if item == 'chihuahua shorthair':
        breed_remove_mix[index] = 'chihuahua'
    elif item == 'collie smooth':
        breed_remove_mix[index] = 'collie'
    elif item == 'anatol shepherd':
        breed_remove_mix[index] = 'anatolian shepherd dog'
    elif item == 'port water dog':
        breed_remove_mix[index] = 'portuguese water dog'
    elif item == 'flat coat retriever':
        breed_remove_mix[index] = 'flat-coated retriever'
    elif item == 'pbgv':
        breed_remove_mix[index] = 'petit basset griffon vendeen'
    elif item == 'bruss griffon':
        breed_remove_mix[index] = 'brussels griffon'
    elif item == 'bluetick hound':
        breed_remove_mix[index] = 'bluetick coonhound'
    elif item == 'wire hair fox terrier':
        breed_remove_mix[index] = 'wire fox terrier'
    elif item == 'dachshund wirehair':
        breed_remove_mix[index] = 'dachshund'
    elif item == 'rhod ridgeback':
        breed_remove_mix[index] = 'rhodesian ridgeback'
    elif item == 'picardy sheepdog':
        breed_remove_mix[index] = 'berger picard'
    elif item == 'st. bernard rough coat':
        breed_remove_mix[index] = 'st. bernard'
    elif item == 'old english bulldog':
        breed_remove_mix[index] = 'olde english bulldogge'
    elif item == 'english bulldog':
        breed_remove_mix[index] = 'olde english bulldogge'
    elif item == 'chesa bay retr':
        breed_remove_mix[index] = 'chesapeake bay retriever'
    elif item == 'dachshund longhair':
        breed_remove_mix[index] = 'dachshund'
    elif item == 'chihuahua longhair':
        breed_remove_mix[index] = 'chihuahua'
    elif item == 'chinese sharpei':
        breed_remove_mix[index] = 'shar-pei'
    elif item == 'standard poodle':
        breed_remove_mix[index] = 'poodle'
    elif item == 'bull terrier miniature':
        breed_remove_mix[index] = 'miniature bull terrier'
    elif item =='st. bernard smooth coat':
        breed_remove_mix[index] = 'st. bernard'
    elif item =='redbone hound':
        breed_remove_mix[index] = 'redbone coonhound'
    elif item == 'cavalier span':
        breed_remove_mix[index] = 'cavalier king charles spaniel'
    elif item == 'collie rough':
        breed_remove_mix[index] = 'collie'
    elif item == 'german shorthair pointer':
        breed_remove_mix[index] = 'german shorthaired pointer'
    elif item == 'english pointer':
        breed_remove_mix[index] = 'pointer'
    elif item == 'mexican hairless':
        breed_remove_mix[index] = 'xoloitzcuintli'
    elif item =='dogo argentino':
        breed_remove_mix[index] = 'argentine dogo'
    elif item == 'queensland heeler':
        breed_remove_mix[index] = 'australian cattle dog'
        
'''breed_remove_mix list now contains all the corrected dog breed name'''
    
'''Based on some analysis, also convert cats into different sizes'''
'''Since there are not that many unique species of cat, it might be faster to compile
size lists manually. The category of size of cats will consist of small, medium and large'''
'''Reference: http://www.petguide.com/breeds/cat/domestic-longhair/
Domestic species are the most popular and abundant and they are of mixed ancestry. Consequently,
they are difficult to define in one size category and I will put them under domestic category'''
small_cat_list = ['munchkin longhair', ]
medium_cat_list = ['exotic shorthair', 'persian', 'abyssinian', 'sphynx', 'siamese',
                   'cornish rex', 'devon rex', 'burmese', 'tonkinese', 'russian blue', 
                   'manx', 'japanese bobtail', 'balinese', 'bombay', 'havana brown',
                   'bengal', 'cymric', 'himalayan', 'snowshoe', 'javanese', 'havana brown', 'angora']
large_cat_list = ['american shorthair', 'british shorthair', 'norwegian forest cat', 'ocicat',
                  'turkish van', 'pixiebob shorthair', 'maine coon', 'ragdoll']
domestic = ['domestic longhair', 'domestic medium hair', 'domestic shorthair']

'''Compare the breed in the data to crawled size lists and create a list indicating
the size of each breed. By zipping two lists, we can also ignore species that
belong to cats. Some breed names were manually modified to fit the crawled list'''
type_list = list(data_train['AnimalType'])

convert_list, excluded_list_dog, mix_breed_list_dog, cat_breed_list = convert_breed(type_list, 
                                     breed_remove_mix, small_dog_compile, 
                                     medium_dog_compile, large_dog_compile, giant_dog_compile,
                                     small_cat_list, medium_cat_list, large_cat_list, domestic)

'''Some animals came out excluded. Let us find out which species and add their information manually.
This list was used to make corrections for dog breed name in the above code. It should now only
contain unknown values.'''
excluded_unique_dog = list(set(excluded_list_dog))

'''Whew finally add the cleansed convert_dog_list to our data_train DataFrame'''
data_train['Size'] = convert_list

'''Now let us apply the same size conversion for cats as well. First find out how many unique cat breeds
are present from the cat_breed_list derived from convert_breed_dog function. This list was used to manually
construct the conversion lists for cats in the above code.'''
cat_unique = list(set(cat_breed_list))

'''Confirm that only expected values small, medium, large, giant, unknown and domestic
are present in the Size column'''
size_check_list = list(data_train['Size'].unique())

'''Before we proceed let us see how result varies with respect to the size column'''
plt.figure()
sns.factorplot(x = 'AnimalType', hue = 'Size', col = 'OutcomeType', data = data_train, kind = 'count')

'''Another preprocessing portion

For breed -> Create 2 columns -> main_breed, sub_breed
There are some species with 3 mixes, but there are only 10 of them so we'll
just consider main and sub breeds.

For color -> Create 2 columns -> main_color, sub_color
'''

'''Reference: https://stackoverflow.com/questions/2600191/how-can-i-count-the-occurrences-of-a-list-item-in-python
Useful reference for counting number of occurences'''
check_breed = list(data_train['Breed'])
count = 0
for item in check_breed:
    if item.count('/') == 2:
        count += 1
'''There are only 10, so we will just ignore the third sub breed'''

'''First create columns corresponding to main and sub breeds'''
main_breed, sub_breed = breed_separator(breed_remove_mix)
data_train['Main_Breed'], data_train['Sub_Breed'] = main_breed, sub_breed

'''Then create columns corresponding to main and sub colors'''
color_list = list(data_train['Color'])
main_color, sub_color = color_separator(color_list)
data_train['Main_Color'], data_train['Sub_Color'] = main_color, sub_color

'''Tried another strategy with how to separate. Basically, mixed breeds are labeled
as mix.'''
pure_mix_compile = breed_separator_mix(breed_remove_mix)
data_train['Mix_Breed'] = pure_mix_compile

'''Let us make a copy of data_train in case we need further processing'''
data_train_le = data_train.copy()

'''The get_dummies method is mentioned in Python Learning by Sebastian Raschka.
It is much more convenient than one hot encoding IMO. This only works for strings btw.'''
main_color_le = pd.get_dummies(data_train_le[['Main_Color']])
sub_color_le = pd.get_dummies(data_train_le[['Sub_Color']])
main_breed_le = pd.get_dummies(data_train_le[['Main_Breed']])
sub_breed_le = pd.get_dummies(data_train_le[['Sub_Breed']])
year_le = pd.get_dummies(data_train_le[['OutcomeYearstr']])
month_le = pd.get_dummies(data_train_le[['OutcomeMonthstr']])
day_le = pd.get_dummies(data_train_le[['OutcomeDaystr']])
hour_le = pd.get_dummies(data_train_le[['OutcomeHourstr']])
size_le = pd.get_dummies(data_train['Size'])
sex_le = pd.get_dummies(data_inter['SexuponOutcome'])
season_le = pd.get_dummies(data_train_le[['OutcomeSeason']])


###############################################################################
'''The preprocessing portion is complete for now. Codes below will be implementation
of various decision tree based algorithms such as random forest, adaboost, XGboost'''
###############################################################################
'''Let us first run a simple case using RandomForest'''
dfX = data_train_le[['Name', 'AnimalType', 'AgeuponOutcome']]
dfX = pd.concat([dfX, size_le, main_breed_le, sub_breed_le, main_color_le, sub_color_le, year_le, month_le, sex_le], axis = 1)
dfY = data_train_le['OutcomeType']

dfX_train, dfX_test, dfY_train, dfY_test = train_test_split(dfX, dfY, test_size = 0.2, random_state=0)

'''Fuck it let us give Xgboost a try'''
model_xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=2, nthread=-1)
model_xgb.fit(dfX_train, dfY_train)

'''Train set accuracy'''
y_pred_train = model_xgb.predict(dfX_train)
print(classification_report(dfY_train, y_pred_train))

'Test set accuracy'''
y_pred_test = model_xgb.predict(dfX_test)
print(classification_report(dfY_test, y_pred_test))

'''Reference:https://www.kaggle.com/c/shelter-animal-outcomes/discussion/22119'''
'''This reference mentions the data exploit and how it affected the result'''
y_proba = model_xgb.predict_proba(dfX_test)
performance = log_loss(dfY_test, y_proba)

###############################################################################
'''2nd data formatting'''
dfX2 = data_train_le[['Name', 'AnimalType', 'AgeuponOutcome']]
dfX2 = pd.concat([dfX2, main_breed_le, main_color_le, month_le, sex_le], axis = 1)
dfY2 = data_train_le['OutcomeType']

dfX2_train, dfX2_test, dfY2_train, dfY2_test = train_test_split(dfX2, dfY2, test_size = 0.2, random_state=0)

'''Fuck it let us give Xgboost a try'''
model_xgb2 = xgboost.XGBClassifier(n_estimators=100, max_depth=2, nthread=-1)
model_xgb2.fit(dfX2_train, dfY2_train)

'''Train set accuracy'''
y_pred_train2 = model_xgb2.predict(dfX2_train)
print(classification_report(dfY2_train, y_pred_train2))

'Test set accuracy'''
y_pred_test2 = model_xgb2.predict(dfX2_test)
print(classification_report(dfY2_test, y_pred_test2))

'''Reference:https://www.kaggle.com/c/shelter-animal-outcomes/discussion/22119'''
'''This reference mentions the data exploit and how it affected the result'''
y_proba2 = model_xgb2.predict_proba(dfX2_test)
performance2 = log_loss(dfY2_test, y_proba2)

###############################################################################
'''3rd data formatting'''
dfX3 = data_train_le[['Name', 'AnimalType', 'AgeuponOutcome']]
dfX3 = pd.concat([dfX3, main_breed_le, sub_breed_le, main_color_le, sub_color_le, month_le, sex_le], axis = 1)
dfY3 = data_train_le['OutcomeType']

dfX3_train, dfX3_test, dfY3_train, dfY3_test = train_test_split(dfX3, dfY3, test_size = 0.2, random_state=0)

'''Fuck it let us give Xgboost a try'''
model_xgb3 = xgboost.XGBClassifier(n_estimators=100, max_depth=2, nthread=-1)
model_xgb3.fit(dfX3_train, dfY3_train)

'''Train set accuracy'''
y_pred_train3 = model_xgb3.predict(dfX3_train)
print(classification_report(dfY3_train, y_pred_train3))

'Test set accuracy'''
y_pred_test3 = model_xgb3.predict(dfX3_test)
print(classification_report(dfY3_test, y_pred_test3))

'''Reference:https://www.kaggle.com/c/shelter-animal-outcomes/discussion/22119'''
'''This reference mentions the data exploit and how it affected the result'''
y_proba3 = model_xgb3.predict_proba(dfX3_test)
performance3 = log_loss(dfY3_test, y_proba3)

###############################################################################
'''Investigate the effect of name'''
'''3rd data formatting'''
dfX4 = data_train_le[['Name', 'AnimalType', 'AgeuponOutcome']]
dfX4 = pd.concat([dfX4, size_le, main_color_le, sub_color_le, month_le, year_le, sex_le], axis = 1)
dfY4 = data_train_le['OutcomeType']

dfX4_train, dfX4_test, dfY4_train, dfY4_test = train_test_split(dfX4, dfY4, test_size = 0.2, random_state=0)

'''Fuck it let us give Xgboost a try'''
model_xgb4 = xgboost.XGBClassifier(n_estimators=100, max_depth=2, nthread=-1)
model_xgb4.fit(dfX4_train, dfY4_train)

'''Train set accuracy'''
y_pred_train4 = model_xgb4.predict(dfX4_train)
print(classification_report(dfY4_train, y_pred_train4))

'Test set accuracy'''
y_pred_test4 = model_xgb4.predict(dfX4_test)
print(classification_report(dfY4_test, y_pred_test4))

'''Reference:https://www.kaggle.com/c/shelter-animal-outcomes/discussion/22119'''
'''This reference mentions the data exploit and how it affected the result'''
y_proba4 = model_xgb4.predict_proba(dfX4_test)
performance4 = log_loss(dfY4_test, y_proba4)

###############################################################################
'''Investigate the effect of name'''
'''3rd data formatting'''
dfX5 = data_train_le[['Name', 'AnimalType', 'AgeuponOutcome']]
dfX5 = pd.concat([dfX5, main_breed_le, sub_breed_le, main_color_le, sub_color_le, season_le, sex_le], axis = 1)
dfY5 = data_train_le['OutcomeType']

dfX5_train, dfX5_test, dfY5_train, dfY5_test = train_test_split(dfX5, dfY5, test_size = 0.2, random_state=0)

'''Fuck it let us give Xgboost a try'''
model_xgb5 = xgboost.XGBClassifier(n_estimators=100, max_depth=2, nthread=-1)
model_xgb5.fit(dfX5_train, dfY5_train)

'''Train set accuracy'''
y_pred_train5 = model_xgb5.predict(dfX5_train)
print(classification_report(dfY5_train, y_pred_train5))

'Test set accuracy'''
y_pred_test5 = model_xgb5.predict(dfX5_test)
print(classification_report(dfY5_test, y_pred_test5))

'''Reference:https://www.kaggle.com/c/shelter-animal-outcomes/discussion/22119'''
'''This reference mentions the data exploit and how it affected the result'''
y_proba5 = model_xgb5.predict_proba(dfX5_test)
performance5 = log_loss(dfY5_test, y_proba5)

###############################################################################
'''Include day and hour information as well'''
dfX6 = data_train_le[['Name', 'AnimalType', 'AgeuponOutcome']]
dfX6 = pd.concat([dfX6, size_le, main_breed_le, sub_breed_le, main_color_le, sub_color_le, 
                 year_le, month_le, day_le, hour_le, sex_le], axis = 1)
dfY6 = data_train_le['OutcomeType']

dfX6_train, dfX6_test, dfY6_train, dfY6_test = train_test_split(dfX6, dfY6, test_size = 0.2, random_state=0)

'''Fuck it let us give Xgboost a try'''
model_xgb6 = xgboost.XGBClassifier(n_estimators=100, max_depth=2, nthread=-1)
model_xgb6.fit(dfX6_train, dfY6_train)

'''Train set accuracy'''
y_pred_train6 = model_xgb6.predict(dfX6_train)
print(classification_report(dfY6_train, y_pred_train6))

'Test set accuracy'''
y_pred_test6 = model_xgb6.predict(dfX6_test)
print(classification_report(dfY6_test, y_pred_test6))

'''Reference:https://www.kaggle.com/c/shelter-animal-outcomes/discussion/22119'''
'''This reference mentions the data exploit and how it affected the result'''
y_proba6 = model_xgb6.predict_proba(dfX6_test)
performance6 = log_loss(dfY6_test, y_proba6)

###############################################################################
'''Include day and hour information as well'''
dfX7 = data_train_le[['Name', 'AnimalType', 'AgeuponOutcome']]
dfX7 = pd.concat([dfX7, size_le, main_breed_le, sub_breed_le, main_color_le, sub_color_le, 
                 year_le, month_le, day_le, hour_le, sex_le], axis = 1)
dfY7 = data_train_le['OutcomeType']

dfX7_train, dfX7_test, dfY7_train, dfY7_test = train_test_split(dfX7, dfY7, test_size = 0.2, random_state=0)

'''Fuck it let us give Xgboost a try'''
model_xgb7 = xgboost.XGBClassifier(n_estimators=100, max_depth=8, nthread=-1)
model_xgb7.fit(dfX7_train, dfY7_train)

'''Train set accuracy'''
y_pred_train7 = model_xgb7.predict(dfX7_train)
print(classification_report(dfY7_train, y_pred_train7))

'''Test set accuracy'''
y_pred_test7 = model_xgb7.predict(dfX7_test)
print(classification_report(dfY7_test, y_pred_test7))

'''Reference:https://www.kaggle.com/c/shelter-animal-outcomes/discussion/22119'''
'''This reference mentions the data exploit and how it affected the result'''
y_proba7 = model_xgb7.predict_proba(dfX7_test)
performance7 = log_loss(dfY7_test, y_proba7)

###############################################################################
'''Include day and hour information as well'''
dfX8 = data_train_le[['Name', 'AnimalType', 'AgeuponOutcome']]
dfX8 = pd.concat([dfX8, size_le, main_breed_le, sub_breed_le, main_color_le, sub_color_le, 
                 year_le, month_le, day_le, hour_le, sex_le], axis = 1)
dfY8 = data_train_le['OutcomeType']

dfX8_train, dfX8_test, dfY8_train, dfY8_test = train_test_split(dfX8, dfY8, test_size = 0.2, random_state=0)

'''Fuck it let us give Xgboost a try'''
model_xgb8 = xgboost.XGBClassifier(n_estimators=300, max_depth=8, nthread=-1)
model_xgb8.fit(dfX8_train, dfY8_train)

'''Train set accuracy'''
y_pred_train8 = model_xgb8.predict(dfX8_train)
print(classification_report(dfY8_train, y_pred_train8))

'''Test set accuracy'''
y_pred_test8 = model_xgb8.predict(dfX8_test)
print(classification_report(dfY8_test, y_pred_test8))

'''Reference:https://www.kaggle.com/c/shelter-animal-outcomes/discussion/22119'''
'''This reference mentions the data exploit and how it affected the result'''
y_proba8 = model_xgb8.predict_proba(dfX8_test)
performance8 = log_loss(dfY8_test, y_proba8)

##############################################################
'''Feature importance analysis code'''
'''Reference: Python Machine Learning'''
feat_labels = np.array(dfX8.columns)
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs = -1)
forest.fit(dfX8_train, dfY8_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
indices = indices[:30]

for f in range(len(indices)):
    print("%2d %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]] ))

plt.figure()
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), feat_labels[indices], rotation=90)
plt.tight_layout()
plt.show()

###############################################################################
'''Let us try and merge train.csv data withe intake data from Austin website'''