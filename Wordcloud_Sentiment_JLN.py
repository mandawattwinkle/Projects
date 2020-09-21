# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:26:49 2020

@author: Asus
"""

# analyzing the s entiment of the speech and the word cloud together

# load the libraries

from wordcloud import WordCloud, STOPWORDS

import pandas as pd

from textblob import TextBlob

#Load data

dataset = pd.read_csv('JLN_freedom.txt')



#Converting data into string format

dataset = dataset.to_string(index = False) 

type(dataset)

# result before cleaning

b1 =TextBlob(dataset)

print(b1.sentiment)  # 16 % positive  
# you will see the diff after preping the data



#-------------------Cleaning the data-----------------------------------

import re
data = dataset.replace('NaN','')

dataset = re.sub("[^A-Za-z0-9]+"," ",data)
dataset



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









#WORDCLOUD
#=============================================================================



stopword = set(STOPWORDS)

    

wc = WordCloud(width = 800, height = 800, 

                   background_color="White",

                   mask=None,

                   max_words=150,

                   stopwords=stopword,

                   min_font_size = 10).generate(dataset)

wc.to_file("WC_data.png")
   


    