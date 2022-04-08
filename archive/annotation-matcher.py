#!/usr/bin/python
# -*- coding: utf-8 -*-

#=================================================#
#
# Description: 
# A Python script to join the annotated dataset of oral questions (Comparative Agendas scheme) 
# to the original text of those questions in the Lipad digitized Hansard.
#
# Usage: 
# python annotation-matcher.py hansard-file annotated-file 
#
# The script will write hansard-file_annotated.csv in the working dir, a copy of 
# hansard-file containing new columns: 'annotated' for the main topics of 
# the comparative agendas project and 'subannotated' for the subtopics.
#
# Note (1): hansard-file is a dataset in Lipad format (www.lipad.ca) 
# for the years 1983 to 2004, subset on oral questions.
#  
# Note (2): Both the Hansard and annotated files must be ordered chronologically.
# If the ordering of rows is modified, the script will likely not work properly. 
#
# @author: L. Rheault
# 
#=================================================#

import os, sys
import re
import pandas as pd
from unidecode import unidecode
import unicodedata

def noAccents(x):
	x = unicode(x)
	x = unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore')
	x = x.encode('utf-8')
	return x

pathHansard = str(sys.argv[1]) 
pathOQ = str(sys.argv[2])
a = pd.read_table(pathHansard,delimiter=",",header=0,dtype=object,encoding='utf-8')
b = pd.read_table(pathOQ, delimiter=",",header=0,dtype=object,encoding='utf-8')
a['n'] = a.apply(lambda row: noAccents(row['speakername']), axis=1)
#The new columns with annotations.
a['annotated'] = ""
a['subannotated'] = ""
# Standardizing the dates.
a['dt'] = pd.Series([pd.to_datetime(date) for date in a.speechdate])
b['dt'] = pd.to_datetime(b.Year+b.Month+b.Day,format='%Y%m%d')

for date, group in a.groupby('dt'):
	counter = []
	for i, lines in a[a.dt==date].iterrows():
		if i==0 or a.subtopic.loc[i]!=a.subtopic.loc[i-1]:
			for j, rows in b[b.dt==date].iterrows():
				if b.Question1_LastName.loc[j] in a.n.loc[i] and j not in counter:
					a.ix[a.index==i,'annotated'] = b.Topic_1.loc[j]
					a.ix[a.index==i,'subannotated'] = b.Subtopic_1.loc[j]
					counter.append(j)
		else:
			a.ix[a.index==i,'annotated'] = a['annotated'].loc[i-1]
			a.ix[a.index==i,'subannotated'] = a['subannotated'].loc[i-1]					

a = a.drop(['dt','n'], axis=1) 
a.to_csv(str(sys.argv[1])+"_annotated.csv",delimiter=",",index=False,encoding='utf-8')
