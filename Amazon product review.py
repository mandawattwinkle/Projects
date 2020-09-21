# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:52:47 2020

@author: Asus
"""
#Import Library

import pandas as pd

from textblob import TextBlob


#Load data

dataset = pd.read_csv('JLN_freedom.txt')


#Converting data into string format

dataset = dataset.to_string(index = False) 

type(dataset)



b1 =TextBlob(dataset)

print(b1.sentiment)  # 15.79% positive only 
# you will see the diff after preping the data



#-------------------Cleaning the data-----------------------------------

import re

dataset = re.sub("[^A-Za-z0-9]+"," ",dataset)



#----------------------Tokenization--------------------------------------------

import nltk

#nltk.download()
# use the above command only after you have installed nltk frshly


#for word in dataset[:500]:

    #print(word, sep='',end='')

    

from nltk.tokenize import word_tokenize

Tokens = word_tokenize(dataset)

#print (Tokens)



#No. of tokens in the dataset

len(Tokens)



#Freq of occurence of distinct elements

from nltk.probability import FreqDist

fdist = FreqDist()



for word in Tokens:

    fdist[word.lower()]+=1

fdist

fdist.plot(20)



#-------------------------Stemming----------------------------------------

from nltk.stem import PorterStemmer

pst=PorterStemmer()

pst.stem("having")





#-------------Remove the Stop Words---------------------

import nltk.corpus



#Enlisting the stopwords present in English lang

stopwords = nltk.corpus.stopwords.words('english')

stopwords[0:10]



#Getting rid of stopwords

for FinalWord in Tokens:

    if FinalWord not in stopwords:

        print(FinalWord)

        

#Classification of words as Positive, Negative & Neutral



#Calculating final Sentiment Score

b2 =TextBlob(FinalWord)

print(b2.sentiment)
