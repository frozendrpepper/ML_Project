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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
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
    year_list, month_list = [], []
    for item in datetime:
        year_list.append(item[:4])
        month_list.append(item[5:7])
    return year_list, month_list
    
def datetime_converter(data_train):
    '''Convert date time into year and month'''
    datetime = list(data_train['DateTime'])
    year_list, month_list = [], []
    for item in datetime:
        year_list.append(int(item[:4]))
        month_list.append(int(item[5:7]))
    return year_list, month_list

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
                breed_compile.append('small')
            elif breed in medium_cat_list:
                breed_compile.append('medium')
            elif breed in large_cat_list:
                breed_compile.append('large')
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
            sub_breed.append(0)
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
            sub_color.append(0)
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
year_list, month_list = datetime_str_converter(data_inter)
date_list = []
for year, month in zip(year_list, month_list):
    date_list.append(int(year + month))
data_inter['Date'] = date_list
year_list, month_list = datetime_converter(data_inter)
data_inter['OutcomeYear'], data_inter['OutcomeMonth'] = year_list, month_list
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
'''Make columns in both string an integer format as this will be useful later on'''
year_list, month_list = datetime_str_converter(data_train)
data_train['OutcomeYearstr'], data_train['OutcomeMonthstr'] = year_list, month_list
date_list = []
for year, month in zip(year_list, month_list):
    date_list.append(int(year + month))
data_train['Date'] = date_list

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

'''Let us first apply label encoding for all the string valued columns'''
size_mapping = {'unknown': 0, 'small':1, 'medium':2, 'giant':3, 'large':4, 'domestic':5, 'mix':6}
data_train_le = data_train.copy()
data_train_le['Size'] = data_train_le['Size'].map(size_mapping)

'''The get_dummies method is mentioned in Python Learning by Sebastian Raschka.
It is much more convenient than one hot encoding IMO. This only works for strings btw.'''
main_color_le = pd.get_dummies(data_train_le[['Main_Color']])
main_breed_le = pd.get_dummies(data_train_le[['Main_Breed']])
year_le = pd.get_dummies(data_train_le[['OutcomeYearstr']])
month_le = pd.get_dummies(data_train_le[['OutcomeMonthstr']])

###############################################################################
'''The preprocessing portion is complete for now. Codes below will be implementation
of various decision tree based algorithms such as random forest, adaboost, XGboost'''
###############################################################################
'''Let us first run a simple case using RandomForest'''
dfX = data_train_le[['Name', 'OutcomeSubtype', 'AnimalType', 'SexuponOutcome', 
                     'AgeuponOutcome', 'Size']]
dfX = pd.concat([dfX, main_color_le, year_le, month_le], axis = 1)
dfY = data_train_le['OutcomeType']

'''Fuck it let us give Xgboost a try'''
model_xgb = xgboost.XGBClassifier(n_estimators=1000, max_depth=3)
model_xgb.fit(dfX, dfY)

y_pred = model_xgb.predict(dfX)
print(classification_report(dfY, y_pred))