#!/usr/bin/python
# -*- coding: utf-8 -*-

#=====================================================================#
# 
# Description: 
# A script to evaluate the anxiety classifier on a corpus of annotated Hansard sentences.    
#
# Usage: 
# python anxiety-classifier.py [anx/bow/metadata] C
#
# --option anx: anxiety score alone.
# --option bow: anxiety score + BoW (700 best features)
# --option metadata: anxiety score + BoW + metadata (Hansard titles and speaker ID categorical variable.)
# --C = sklearn penalty parameter.
# Dependencies: cf.csv
# 
# The script will display accuracy statistics via standard output.
#
# Results reported in text are obtained using:
# python anxiety-classifier.py anx 0.5
# python anxiety-classifier.py bow 0.5
# python anxiety-classifier.py metadata 0.5
#
# Author: L. Rheault
# 
#=====================================================================#

from __future__ import division
from operator import itemgetter
import os, sys, codecs, re
import pandas as pd
import numpy as np
from time import time
from scipy import sparse
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import StratifiedKFold,StratifiedShuffleSplit,KFold,ShuffleSplit,train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC

#=====================================================================#
# Feature classes.

class anxietyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        anxiety = np.asarray(X['anxiety'].apply(pd.to_numeric))
        return [{'anxiety': z} for z in anxiety]
    def get_feature_names(self):
        return 'anxiety_index'

class presentTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        presents = np.asarray(X['present'].apply(pd.to_numeric))
        return [{'present': z} for z in presents]
    def get_feature_names(self):
        return 'present_tense'

class pastTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        pasts = np.asarray(X['past'].apply(pd.to_numeric))
        return [{'past': z} for z in pasts]
    def get_feature_names(self):
        return 'past_tense'

class futureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        futures = np.asarray(X['future'].apply(pd.to_numeric))
        return [{'future': z} for z in futures]
    def get_feature_names(self):
        return 'future_tense'

class conditionalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        conds = np.asarray(X['conditional'].apply(pd.to_numeric))
        return [{'conditional': z} for z in conds]
    def get_feature_names(self):
        return 'conditional_form'

class futureAnxietyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        futAnx = np.asarray(X['futanx'].apply(pd.to_numeric))
        return [{'futanx': z} for z in futAnx]

class conditionalAnxietyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        condAnx = np.asarray(X['condanx'].apply(pd.to_numeric))
        return [{'condanx': z} for z in condAnx]

class partyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        parties = np.asarray(X['party'].astype('category'))
        return [{'party': party} for party in parties]
    def get_feature_names(self):
        return 'party_name'

class govStatusTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        govS = np.asarray(X['gov'].astype('category'))
        return [{'gov': gov} for gov in govS]
    def get_feature_names(self):
        return 'gov_opp'

class genderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        genders = np.asarray(X['gender'].astype('category'))
        return [{'gender': g} for g in genders]
    def get_feature_names(self):
        return 'gender'

class ageTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        ages = np.asarray(X['age'].apply(pd.to_numeric))
        return [{'age': z} for z in ages]
    def get_feature_names(self):
        return 'age'

class opidTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        opids = np.asarray(X['opid'].astype('category'))
        return [{'opid': z} for z in opids]
    def get_feature_names(self):
        return 'opid'

class hansardTopicTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        text = X['maintopic'].apply(lambda x: x.lower())    
        text = text.tolist()
        return text
    
class hansardSubtopicTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        text = X['subtopic'].apply(lambda x: x.lower())    
        text = text.tolist()
        return text
  
class sentenceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        text = X['sentence'].tolist()
        return text
    
#=====================================================================#
# Accuracy functions.

def simpleAccuracy(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    pcp = metrics.accuracy_score(y_test,pred)
    f1 = metrics.f1_score(y_test,pred)
    mc = metrics.classification_report(y_test, pred)
    cm = metrics.confusion_matrix(y_test, pred)        
    return (float(pcp), float(f1), mc, cm) 

def shuffledAccuracy(X, y, clf):
    pcp = []
    f1 = []
    skf = StratifiedShuffleSplit(y, 3, test_size=0.4, random_state=42)
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        t_pcp = metrics.accuracy_score(y_test,pred)
        t_f1 = metrics.f1_score(y_test,pred)
        pcp.append(t_pcp)
        f1.append(t_f1)        
    return (float(np.mean(pcp)), float(np.mean(f1))) 

def accuracy3Fold(X, y, clf):
    pcp = []
    f1 = []
    k_fold = KFold(a.shape[0], 3, shuffle=True, random_state=42)
    for k, (train, test) in enumerate(k_fold):
        clf.fit(X[train], y[train])
        pred = clf.predict(X[test])
        t_pcp = metrics.accuracy_score(y[test],pred)
        t_f1 = metrics.f1_score(y[test],pred)
        pcp.append(t_pcp)
        f1.append(t_f1)
    return (float(np.mean(pcp)), float(np.mean(f1))) 

#=====================================================================#

if __name__=="__main__":

    t0 = time()    
    print "Opening and preparing the corpus."
    a = pd.read_table("cf.csv",delimiter=",",header=0,dtype=object,encoding="utf-8")

    a['anxclass'] = a.anxclass.apply(pd.to_numeric)
    a['anxiety'] = a.anxiety.apply(pd.to_numeric) 
    a['anxiety'] = a.anxiety+1  # Converted to positive range for computation.
    a['future'] = a.future.apply(pd.to_numeric)  # Tested feature, but unused in final model.
    a['present'] = a.future.apply(pd.to_numeric)  # Tested feature, but unused in final model.
    a['past'] = a.future.apply(pd.to_numeric)  # Tested feature, but unused in final model.
    a['age'] = a.future.apply(pd.to_numeric)    # Tested feature, but unused in final model.
    a['conditional'] = a.conditional.apply(pd.to_numeric)  # Tested feature, but unused in final model.
    a['futanx'] = a['future'] * a['anxiety']  # Tested feature, but unused in final model.
    a['condanx'] = a['conditional'] * a['anxiety']  # Tested feature, but unused in final model.

    X = a[['sentence','anxiety','future','present','past','conditional','futanx','condanx','party','gov','age','gender','opid','maintopic','subtopic']]
    y = a['anxclass'].values.ravel()

    anxietySent = Pipeline([
        ('anxiety', anxietyTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    futureSent = Pipeline([
        ('future', futureTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    presentSent = Pipeline([
        ('present', presentTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    pastSent = Pipeline([
        ('past', pastTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    conditionalSent = Pipeline([
        ('conditional', conditionalTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    futureAnxietySent = Pipeline([
        ('futanx', futureAnxietyTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    conditionalAnxietySent = Pipeline([
        ('condanx', conditionalAnxietyTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    speechParty = Pipeline([
        ('party', partyTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    speechGovStatus = Pipeline([
        ('govS', govStatusTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    genderStatus = Pipeline([
        ('gender', genderTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    ageStatus = Pipeline([
        ('age', ageTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    opidStatus = Pipeline([
        ('opid', opidTransformer()),
        ('dict-vect', DictVectorizer())
        ])

    hansardTopics = Pipeline([
        ('mtopic', hansardTopicTransformer()),
        ('tfidf', TfidfVectorizer(stop_words='english',ngram_range=(1,3)))   
        ])

    hansardSubtopics = Pipeline([
        ('stopic', hansardSubtopicTransformer()),
        ('tfidf', TfidfVectorizer(stop_words='english',ngram_range=(1,3)))   
        ])

    mainText = Pipeline([
        ('text', sentenceTransformer()),
        ('tfidf', TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=1000))   
        ])

    print "Combining features..."
    if str(sys.argv[1])=='anx':
        features = FeatureUnion([('anxietySent', anxietySent)])
        X = features.fit_transform(X)
    elif str(sys.argv[1])=='bow':
        features = FeatureUnion([
                        ('anxietySent', anxietySent),
                        ('mainText',  mainText)])
        X = features.fit_transform(X)
        ch2 = SelectKBest(chi2, k = 700)
        X = ch2.fit_transform(X, y)
    elif str(sys.argv[1])=='metadata':
        features = FeatureUnion([
            ('anxietySent', anxietySent), 
            ('hansardTopics', hansardTopics),
            ('hansardSubtopics', hansardSubtopics),
            ('opidStatus', opidStatus),
            ('mainText',  mainText)
        ])
        X = features.fit_transform(X)
        ch2 = SelectKBest(chi2, k = 700)
        X = ch2.fit_transform(X, y)
    else:
        sys.exit("Please choose option anx, bow or metadata.")

    c = float(sys.argv[2])
    clf = LinearSVC(C=c)
    print "Now Computing models..."

    pcp, f1, mc, cm = simpleAccuracy(X, y, clf)
    print "-----------------------------------------------------------" 
    print "Support Vector Machine (C = %s), random Train/Test." % c
    print "-----------------------------------------------------------" 
    print "The percent correctly predicted is %0.3f" %pcp
    print "The F1-score is %0.3f" %f1
    print mc
    print cm
    print "-----------------------------------------------------------" 

    pcp, f1 = shuffledAccuracy(X, y, clf)
    print "-----------------------------------------------------------" 
    print "Support Vector Machine (C = %s), shuffled Train/Test." % c
    print "-----------------------------------------------------------" 
    print "The percent correctly predicted is %0.3f" %pcp
    print "The F1-score is %0.3f" %f1
    print "-----------------------------------------------------------" 

    pcp, f1 = accuracy3Fold(X, y, clf)
    print "-----------------------------------------------------------" 
    print "Support Vector Machine (C = %s), three-fold." % c
    print "-----------------------------------------------------------" 
    print "The percent correctly predicted is %0.3f" %pcp
    print "The F1-score is %0.3f" %f1
    print "-----------------------------------------------------------" 

    print "Done in %0.3fs" % (time() - t0)
