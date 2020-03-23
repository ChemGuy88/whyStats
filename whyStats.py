#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Notes:

    1. Accuracy might be low because this is a high dimensional problem. Our input is high dimensional , with only 80 cases, but over 100 features when using tfidf

    References:

    1. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
'''

import json, nltk, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from IPython import get_ipython
from importlib import reload
from nltk.corpus import words as nltkWords
from random import sample, shuffle
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle as skshuffle

try:
    reload(funcs)
except:
    t, v, cb = sys.exc_info()
    exceptionName = t.__name__
    exceptionVal = v.args[0]
    if exceptionName == 'NameError' and exceptionVal == "name 'funcs' is not defined":
        import funcs
        reload(funcs)
        from funcs import *
    else:
        raise
# from mlFunctions import *

# # For running IPython magic commands (e.g., %matplotlib)
# ipython = get_ipython()
#
# # Display plots inline and change default figure size
# ipython.magic("matplotlib")

'''#####################################################################
###### Universal variables #############################################
########################################################################'''

userDir = '/Users/herman'
workDir = f'{userDir}/Documents/whyStats'

fnames = ['fsu19-01.json',
          'tcc20-03_95.json',
          'tcc20-03_98.json']

'''#####################################################################
###### Workspace #######################################################
########################################################################'''

def script(interactive, verbose, classifier, yType, randomState):
    '''
    Executes script.
    '''

    '''#################### Load and wrangle data ##########################'''
    data, grades = getData(fnames, verbose, interactive)

    ids = list(data.keys())
    X_train_id, X_test_id = train_test_split(ids, random_state=randomState)

    y_train = grades['grade'].reindex(X_train_id)
    y_test = grades['grade'].reindex(X_train_id)
    X_train_text = [ data[id]['msg'] for id in X_train_id ]
    X_test_text = [ data[id]['msg'] for id in X_test_id ]

    '''################# Create features from training data ####################'''
    X_train_tfidf, count_vect, tfidf_transformer = getFeaturesTrain(X_train_text, interactive, verbose)

    '''################## Create features from test data ####################'''
    X_test_tfidf = getFeaturesTest(X_test_text, count_vect, tfidf_transformer, interactive, verbose)

    # return count_vect, tfidf_transformer, X_train_tfidf, X_test_tfidf

    '''
    # map the words from count_vect.get_feature_names() to the values from tfidf_transformer.toarray()
    from whyStats import *
    interactive=False; verbose=1; classifier='linear'; yType='grade'; randomState=1
    count_vect, tfidf_transformer, X_train_tfidf, X_test_tfidf = script(interactive, verbose, classifier, yType, randomState)
    len(count_vect.get_feature_names()) # 782
    words[5:10]
    # ['able', 'about', 'accessible', 'accounting', 'achieve']
    X = X_train_tfidf.toarray()
    X.shape # (70,782)
    X[:2,:10]
    # array([[0.        , 0.        , 0.        , 0.        , 0.        ,
    #     0.        , 0.05972467, 0.        , 0.        , 0.        ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ,
    #     0.10960418, 0.13082206, 0.        , 0.        , 0.        ]])
    # clf = logisticRegression(X, y, reg='l1')
    '''

    '''####################### get y values ############################'''
    y_train, y_test = getY(yType, grades, X_train_id, X_test_id)

    '''####################### Run workflow on classifier ############################'''

    if classifier == 'mnnb':
        clf = MultinomialNB()
    elif classifier == 'svm':
        clf = SGDClassifier(loss='hinge', penalty='l1',
                            alpha=1e-3, random_state=randomState,
                            max_iter=5, tol=None)
    elif classifier == 'linear':
        clf = SGDRegressor(penalty='l1', random_state=randomState)
    else:
        print(f'\nNo valid classifier was chosen')
        return

    return analysis(clf, X_train_tfidf, X_test_tfidf, y_train, y_test, grades, X_test_id, X_test_text, verbose)


'''#####################################################################
###### def main() ######################################################
########################################################################'''

def main():

    # Parse Arguments
    args = sys.argv[1:]
    options = getOptions(args)
    args = list(options.keys())
    if '-interactive' in args:
        interactive = True
    else:
        interactive = False
    if '-verbose' in args:
        verbose = int(options['-verbose'])
    else:
        verbose = 0
    if '-classifier' in args:
        classifier = options['-classifier']
    else:
        classifier = 'mnnb'
    if '-yType' in args:
        yType = options['-yType']
    else:
        yType = 'letterGrade'
    if '-randomState' in args:
        randomState = int(options['-randomState'])
    else:
        randomState = None

    # Run script
    script(interactive, verbose, classifier, yType, randomState)
    print('\nDone\n')

if __name__ == '__main__':
    main()

########################################################################
### End of File ########################################################
########################################################################
