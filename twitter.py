# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:55:59 2018

@author: SWAPNIL
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('train.csv')
"""dataset=dataset.query("class_label<=1")
dataset=dataset[['text','class_label']]"""

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,9271):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
  
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

testdataset=pd.read_csv('test.csv')
corpus1 = []
for k in range(0,1089):
    review = re.sub('[^a-zA-Z]', ' ', testdataset['Tweet'][k])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus1.append(review)
  
    
data_testresult = cv.transform(corpus1).toarray()
result=classifier.predict(data_testresult)
#result=pd.DataFrame(result)

import csv
with open('result.csv','w',newline='') as out_file:
    for i in range(result.shape[0]):
        out =" "
        out+=str(result[i])
        out+="\n"
        out_file.write(out)
        


