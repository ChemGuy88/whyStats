#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DOCSTRING
'''

import json
import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from IPython import get_ipython
from nltk.corpus import words as nltkWords
from random import sample, shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle as skshuffle
# from mlFunctions import *

# # For running IPython magic commands (e.g., %matplotlib)
# ipython = get_ipython()
#
# # Display plots inline and change default figure size
# ipython.magic("matplotlib")

'''#####################################################################
########################################################################
### Universal variables ################################################
########################################################################'''

verbose = 1

userDir = '/Users/herman'
workDir = f'{userDir}/Documents/whyStats'

fnames = ['fsu19-01.json',
          'tcc20-03_95.json',
          'tcc20-03_98.json']

'''#####################################################################
########################################################################
### Workspace ##########################################################
########################################################################'''

'''#################### Load and wrangle data ##########################'''

rawData = {fname : [] for fname in fnames}

for fname in fnames:

    fpath = f'{workDir}/{fname}'
    with open(fpath, 'r') as f:
        text = f.read()
        i = text.find('{')
        text = text[i:]
        d = json.loads(text)

    rawData[fname] = d

data = {}
'''
X = { id : {'msg' : '...',\
            'grade' : '...',\
            'name' : '...'\
            }
    }

'''
invalids = []
for fname in fnames:

    participants = rawData[fname]['participants']

    view = rawData[fname]['view']

    section = fname[:-5]

    for d in participants:
        id = int(d['id'])
        name = d['display_name']
        data[id] = {'name' : name, 'section' : section}

    for d in view:
        try:
            isValid = not(d['deleted'])
        except KeyError:
            isValid = True
        if isValid:
            id = int(d['user_id'])
            message = d['message']
            data[id].update( { 'msg' : message } )
        elif not(isValid):
            invalids.append(d)

if verbose >= 1:
    print(f'The number of invalid dictionaries in the JSON objects is {len(invalids)}. They are stored in the "invalid" variable.')

# Remove teacher comments
data1 = data.copy()
for id, d in data1.items():
    name = d['name'].lower()
    if 'herman' in name and 'autore' in name:
        del data[id]
del data1

# get grades
fpath = f'{workDir}/grades.xlsx'
grades = pd.read_excel(fpath,
                       sheet_name = 'grades',
                       index_col = 1,
                       usecols = ['student', 'id', 'grade', 'letterGrade', 'section'])
grades['grade'] = grades['grade'].astype(float)

# Assert all message ids point have a corresponding grade
data1 = data.copy()
gradesid = grades.index.astype(int)
for id, d in data1.items():
    if id not in gradesid:
        del data[id]
del data1


ids = list(data.keys())
X_train_id, X_test_id = train_test_split(ids)

y_train = grades['grade'].reindex(X_train_id)
y_test = grades['grade'].reindex(X_train_id)
X_train_text = [ data[id]['msg'] for id in X_train_id ]
X_test_text = [ data[id]['msg'] for id in X_test_id ]

'''#####################################################################
### Create features from data
    https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
########################################################################'''

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train_text)

# Runtime feedback
if verbose >= 1:
    print(f'X_train_counts shape:\t\t{X_train_counts.shape}')
    Input = input('\nWould you like to get the frequency of a word? Press return to skip\n')
    if Input == '':
        pass
    elif isinstance(Input, str):
        print(f'\nThe word "{Input}" appears {count_vect.vocabulary_.get(Input)} times.')
elif False:
    print(count_vect.get_feature_names())
    print(X_train_counts.toarray())
    print(X_train_counts.shape)

    for w in count_vect.get_feature_names():
        print(f'{w} | {count_vect.vocabulary_.get(w)}')

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

if verbose >= 1:
    print(f'\nX_train_tfidf shape:\t\t{X_train_tfidf.shape}.')

'''####################### Train NBMN classifier #################################'''

clf = MultinomialNB().fit(X_train_tfidf, grades['letterGrade'].reindex(X_train_id))

X_test_counts = count_vect.transform(X_test_text)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = clf.predict(X_test_tfidf)

names = grades['student'].reindex(X_test_id[:3])
messages = X_test_text[:3]
predictions = predicted[:3]
realGrades = grades['letterGrade'].reindex(X_test_id[:3])
for name, message, prediction, grade in zip(names, messages, predictions, realGrades):
    print(f'\n{name}:\n{message}\nPrediction => {prediction}\nActual => {grade}')
# Accuracy might be low because this is a high dimensional problem. Our input is high dimensional , with only 80 cases, but over 100 features.

'''###################### Evaluate NBMN classifier ###############################'''

# See "Evaluation of the performance on the test set" at https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
acc = np.mean(predicted == grades['letterGrade'].reindex(X_test_id))
print(f'\nThe Naive bayes multinomial classifier has an accuracy of {acc}')
