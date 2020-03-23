#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DOCSTRING
'''

import json, nltk, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from IPython import get_ipython
from nltk.corpus import words as nltkWords
from random import sample, shuffle
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle as skshuffle
from tabulate import tabulate

'''#####################################################################
###### Universal variables #############################################
########################################################################'''

userDir = '/Users/herman'
workDir = f'{userDir}/Documents/whyStats'

fnames = ['fsu19-01.json',
          'tcc20-03_95.json',
          'tcc20-03_98.json']

yType = 'grade'

'''#####################################################################
###### Workspace #######################################################
########################################################################'''

def getOptions(args):
    opts = {}
    for arg in args:
        arg, opt = arg.split('=')
        opts[arg] = opt
    return opts

def script(interactive, verbose, classifier, yType):
    '''
    Executes script.
    '''

    '''#################### Load and wrangle data ##########################'''

def getData(fname, verbose, interactive):
    '''
    Load and wrangle data
    '''
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

    # Assert that all message ids have a corresponding grade
    grades1 = data.copy()
    gradesid = grades.index.astype(int)
    for id, d in grades1.items():
        if id not in gradesid:
            del data[id]
    del grades1

    return data, grades

def getFeaturesTrain(X_train_text, interactive, verbose):
    '''
    Create tf-idf features from training data.
    '''
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train_text)

    # Runtime feedback
    if verbose >= 1:
        print(f'\nX_train_counts shape:\t\t{X_train_counts.shape}')
    if interactive:
        Input = input('\nWould you like to get the frequency of a word? Press return to skip\n')
        if Input == '':
            pass
        elif isinstance(Input, str):
            print(f'\nThe word "{Input}" appears {count_vect.vocabulary_.get(Input)} times.')

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    if verbose >= 1:
        print(f'\nX_train_tfidf shape:\t\t{X_train_tfidf.shape}.')

    return X_train_tfidf, count_vect, tfidf_transformer

def getFeaturesTest(X_test_text, count_vect, tfidf_transformer, interactive, verbose):
    '''
    Create tf-idf features from test data.
    '''
    X_test_counts = count_vect.transform(X_test_text)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return X_test_tfidf

def getY(yType, grades, X_train_id, X_test_id):
    '''
    Set a feature as the respone value, y
    '''
    y_train = grades[yType].reindex(X_train_id)
    y_test = grades[yType].reindex(X_test_id)

    return y_train, y_test

def analysis(clf, X_train_tfidf, X_test_tfidf, y_train, y_test, grades, X_test_id, X_test_text, verbose):
    '''
    Train, test, and evaluate a classifier
    '''

    clfName = clf.__class__.__name__

    '''####################### Train NBMN classifier ############################'''

    # return
    clf.fit(X_train_tfidf, y_train)

    '''####################### Test NBMN classifier ############################'''

    predicted = clf.predict(X_test_tfidf)
    predicted = pd.DataFrame(predicted, index=y_test.index, columns=['grade_pred'])

    '''##################### Evaluate NBMN classifier #######################'''
    # https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

    if verbose >= 1:
        ids = X_test_id[:3]
        names = grades['student'].reindex(ids).values
        messages = X_test_text[:3]
        predictions = predicted.loc[ids].iloc[:,0].values
        realGrades = y_test.loc[ids].values
        for name, message, prediction, grade in zip(names, messages, predictions, realGrades):
            print(f'\n{name}:\n{message}\nPrediction => {prediction}\nActual => {grade}')

    if verbose >= 1:
        if clfName in ['MultinomialNB', 'SGDClassifier']:
            acc = np.mean(predicted.values == y_test.values)
            print(f'\nThe {clfName} model has an accuracy of {acc}')
            print()
            print(metrics.classification_report(y_test, predicted))
        elif clfName in ['SGDRegressor']:
            mse = metrics.mean_squared_error(y_test, predicted)
            mae = metrics.mean_absolute_error(y_test, predicted)
            evs = metrics.explained_variance_score(y_test, predicted)
            r2 = metrics.r2_score(y_test, predicted)
            tmse = 'The MSE is'
            tmae = 'The mean absolute error is'
            tevs = 'The explained variance score is'
            tr2 = 'R^2 score is'
            table = [[tmse, mse],
                     [tmae, mae],
                     [tevs, evs],
                     [tr2 , r2 ]]
            print('\n' + tabulate(table))
            df = pd.concat((y_test, predicted), axis=1)
            print()
            print(df)




    return

########################################################################
### End of File ########################################################
########################################################################
