#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:19:41 2018

@author: arunnemani
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

"""
Intializations
"""
nltk.download('punkt')
categories = None #Indicating all 20 categories are included in dataset
remove = ('headers', 'footers', 'quotes')
RANDOM_STATE = 35

"""
Load Dataset
"""
print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "Full dataset (20 groups)")

newsdata_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=RANDOM_STATE,
                                remove=remove)

newsdata_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=RANDOM_STATE,
                               remove=remove)
print('Data loaded')

# split a training set and a test set
Y_train, Y_test = newsdata_train.target, newsdata_test.target

"""
Text preprocessing
"""
stemmer = PorterStemmer()
def Stem_tokenize(text):
    tokens = TreebankWordTokenizer().tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

"""
Feature extraction
"""
def vectorize_dataset(Vectorizer):
    # Three optiions for vectorizer: TfidfVectorizer (default), HashingVectorizer, CountVectorize
    # Initializations
    Stopwords = 'english'
    lowercase = True
    Analyzer = 'word'
    Ngram = (1,2)
    Strip_accents = 'ascii'
    #min_df = 10
    
    if Vectorizer == "HashingVectorizer":
        vectorizer = HashingVectorizer(tokenizer=Stem_tokenize,
                                       stop_words=Stopwords, lowercase=lowercase,
                                       strip_accents=Strip_accents, analyzer=Analyzer,
                                       ngram_range=Ngram,
                                       alternate_sign=False,
                                       n_features=2 ** 16)
        
        X_train = vectorizer.transform(newsdata_train.data)
        print("Chosen vectorizer: HashingVectorizer")
        print("n_samples: {}, n_features: {}".format(X_train.shape, X_train.shape))
        
    elif Vectorizer == "CountVectorizer":
        vectorizer = CountVectorizer(tokenizer=Stem_tokenize,
                                     stop_words=Stopwords, lowercase=lowercase, 
                                     strip_accents=Strip_accents, analyzer=Analyzer,
                                     ngram_range=Ngram)
        
        X_train = vectorizer.fit_transform(newsdata_train.data)
        print("Chosen vectorizer: CountVectorizer")
        print("n_samples: {}, n_features: {}".format(X_train.shape, X_train.shape))
        
    else:
        vectorizer = TfidfVectorizer(tokenizer=Stem_tokenize,
                                     stop_words=Stopwords, lowercase=lowercase,
                                     strip_accents= Strip_accents, analyzer=Analyzer,
                                     ngram_range= Ngram,
                                     sublinear_tf=True,
                                     use_idf=1, smooth_idf=1)
        
        X_train = vectorizer.fit_transform(newsdata_train.data)
        print("Chosen vectorizer: TfidfVectorizer")
        print("n_samples: {}, n_features: {}".format(X_train.shape, X_train.shape))

    
    print("Extracting features from the test data using the same vectorizer")
    X_test = vectorizer.transform(newsdata_test.data)
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
    return X_train, X_test

def show_top20_features(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top20 = np.argsort(classifier.coef_[i])[-20:]
        print("%s: %s" % (category, " ".join(feature_names[top20])))
        


"""
Benchmark classifiers and plot
"""
def benchmarkClassifiers(clf):
    print("Training: ")
    print(clf)
    X_train, X_test = vectorize_dataset("HashingVectorizer")
    Y_pred = clf.fit(X_train, Y_train).predict(X_test)
    # Calculate accuracy score
    HashingScore = metrics.accuracy_score(Y_test, Y_pred)
    
    X_train, X_test = vectorize_dataset("CountVectorizer")
    Y_pred = clf.fit(X_train, Y_train).predict(X_test)
    # Calculate accuracy score
    CountScore = metrics.accuracy_score(Y_test, Y_pred)
    
    X_train, X_test = vectorize_dataset("TfidfVectorizer")
    Y_pred = clf.fit(X_train, Y_train).predict(X_test)
    # Calculate accuracy score
    TfidScore = metrics.accuracy_score(Y_test, Y_pred)
    
    print("Training complete")
    clf_descr = str(clf).split('(')[0]
    return clf_descr, HashingScore, CountScore, TfidScore

def plotClassifierResults(results):

    indices = np.arange(len(results))
    results = [[x[i] for x in results] for i in range(4)]
    clf_names, HashingScore, CountScore, TfidScore = results
    plt.title("Accuracy scores")
    plt.bar(indices, HashingScore, .1, label="Hashing Vectorizer", color='orange')
    plt.bar(indices + .1, CountScore, .1, label="Count Vectorizer", color='c')
    plt.bar(indices + .2, TfidScore, .1, label="Tfidf Vectorizer", color='m')
    plt.xticks(indices, clf_names)
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    plt.xticks(rotation=45)
    plt.show()
    
"""
Apply various classifiers
"""
results = []
results.append(benchmarkClassifiers(MultinomialNB(alpha=.01)))
results.append(benchmarkClassifiers(LogisticRegression(C=1000)))
results.append(benchmarkClassifiers(LinearSVC(loss='l2', penalty="l2", dual=False, tol=1e-3, C =1)))
results.append(benchmarkClassifiers(SGDClassifier(alpha=.0001, n_iter=50, penalty="l2")))
results.append(benchmarkClassifiers(RandomForestClassifier(n_estimators=100)))


plotClassifierResults(results)