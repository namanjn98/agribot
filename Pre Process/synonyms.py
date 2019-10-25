#!/usr/bin/env python
# coding: utf-8

# In[20]:


import json,os
from nltk.corpus import wordnet as wn
import pandas as pd 
from nltk.corpus import stopwords 
import numpy as np
from nltk.probability import FreqDist, MLEProbDist
from numpy import array
import string
import operator
import copy
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 

def inputques():
    words = pd.DataFrame()
    tempo = pd.read_csv('all_files.csv')
    words = pd.concat([tempo,words])

    questions = list(words['questions'])
    answers = list(words['answers'])
    district = list(words['district'])
    state = list(words['state'])

    main = []
    for w in list(questions):
        w=str(w)
        main.append(w.split(' '))

    new_words = []
    all_words = []
    for w,i in enumerate(main):
        temp = []
        for j in i:
            if j not in stop_words:
                temp.append(j)
                all_words.append(j)

        new_words.append(temp)
    return [district,state,answers,new_words,all_words]


def TrainingSynonymCheck(new_words,all_words):
    realwords=[]
    for i in all_words:
        if i.isalpha():
            realwords.append(i)
    
    lemmatized_words=[]
    for i in realwords:
        n=lemmatizer.lemmatize(i)
        lemmatized_words.append(str(n))   
    
    FreqDictionary=FreqDist(lemmatized_words)
    l=sorted(FreqDictionary.items(), key=operator.itemgetter(1),reverse=True)
    
    unique_words=[]
    for i in l:
        unique_words.append(i[0])

    synDict={}
    flag=0
    for word in unique_words:
        if word.isalpha():
            syns=wn.synsets(word)
            lst=[word]
            if syns!=[]:
                s=syns[0]
                a=s.lemmas()
                for i in a:
                    f=i.name()
                    if f!=word and (f not in lst):
                        lst.append(str(f))
            if lst!=[]:
                synDict[word]=(lst)
                
    synunique_words=copy.deepcopy(unique_words)         ## synunique_words : copy that contains all the unique words in the questions initially
   
    for word in unique_words:
        i = unique_words.index(word)
        if word in synDict.keys():
            for syn in synDict[word]:            
                for j in unique_words[i+1:]:
                    if syn==j:
                        idx=unique_words.index(j)
                        unique_words[idx]=word
            
    return [unique_words,synunique_words]              # unique words : changed words , synunique_words : original words unique 


def Difference(new_words,all_words):
    lst=TrainingSynonymCheck(new_words,all_words)
    unique_words=lst[0]
    synunique_words=lst[1]
    diff={}
    for i in range(0,len(unique_words)):
        if unique_words[i]!=synunique_words[i]:
            diff[str(synunique_words[i])]=unique_words[i]
    return diff


def changingQuestions(new_words,all_words):
    diff=Difference(new_words,all_words)
    changed_ques=copy.deepcopy(new_words)
    changedindexes=[]
    for ques in changed_ques:
        for word in ques:
            if word in diff.keys():
                i=changed_ques.index(ques)
                changedindexes.append(i)
                idx=ques.index(word)
                ques[idx]=diff[word]    
    return changed_ques


def createcsv():
    [district,state,answers,new_words,all_words]=inputques()
    changed_ques= changingQuestions(new_words,all_words)
    p=len(district)
    q=len(state)
    r=len(answers)
    s=len(changed_ques)
    print p,q,r,s


createcsv()



