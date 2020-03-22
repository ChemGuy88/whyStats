#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DOCSTRING
'''

import nltk
import matplotlib.pyplot as plt
import numpy as np
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

'''
    Generate Dataset
'''

numPos = 20
numNeg = 20
numObs = numPos + numNeg

numWords = 6

true = 'I am smart'
false = 'I will fail'

X = []
for i in range(numPos):
    # words = r.get_random_words(limit=5)
    rWords = sample(nltkWords.words(), numWords)
    X.append([true] + rWords)

for i in range(numNeg):
    rWords = sample(nltkWords.words(), numWords)
    X.append([false] + rWords)

for i in range(numObs):
    wordsList = X[i]
    _ = shuffle(wordsList)
    sentence = ' '.join(wordsList)
    X[i] = sentence

y = [1] * numPos + [0] * numNeg

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

'''
    Tokenize
    https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
'''

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X)

'''
# Look at the tokenized data

print(vectorizer.get_feature_names())
print(X_train_counts.toarray())
print(X_train_counts.shape)

for w in vectorizer.get_feature_names():
    print(f'{w} | {vectorizer.vocabulary_.get(w)}')
'''

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y)

X_new_counts = vectorizer.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(X_test, predicted):
    print('%r => %s' % (doc, category))
