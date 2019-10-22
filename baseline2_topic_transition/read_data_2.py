#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:14:50 2018

@author: liyt
"""
import json
import string
import nltk
import nltk.data
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')




def read_data(name, cor_size):
    # 1.1 Read all 5 stars, restuarant review
    yelp_restaurant_data_all_read = []
    with open('yelp-data/' + name, 'r') as f:
        for line in f:
            try:
                j = line.split('|')[-1]
                yelp_restaurant_data_all_read.append(json.loads(j))
            except ValueError:
                    # You probably have bad JSON
                continue
        
    # 1.3 Extract Reviews
    n = len(yelp_restaurant_data_all_read[0])
    review = list()
    for i in range(n):
        review.append(''.join(yelp_restaurant_data_all_read[0][i]['text']))

    small_yelp = review[0:cor_size]
    
    ## 2.1 Get staistics of text
    max_select_length = 150 # Max length of sentence
    min_select_length = 20 # Min Length of sentence
    max_length = 0
    total_len = 0
    min_len = 10e4 # initial set of minimum length


    for i in range(len(small_yelp)):
        if max_select_length < len(small_yelp[i].split()):
            small_yelp[i] = ' '.join(small_yelp[i].split()[:max_select_length])
        if max_length < len(small_yelp[i].split()):
            max_length = len(small_yelp[i].split())

    text_trans = list()
    for i in range(len(small_yelp)):
        if min_select_length < len(small_yelp[i].split()):
            text_trans.append(small_yelp[i])

    small_yelp = text_trans
    for i in range(len(small_yelp)):
        total_len += len(small_yelp[i].split())
        if len(small_yelp[i].split()) <= min_len:  
            min_len = len(small_yelp[i].split())

    print('Text Max Length: {:04d}, Text Average Length: {:.2f}, Text Min Length: {:04d}'.
          format(max_length, total_len/len(small_yelp), min_len))
    

    n = len(small_yelp)
    l = list() # length of document i
    for i in range(n):
        j = len(nltk.sent_tokenize(small_yelp[i]))
        l.append(j) 
    L = max(l) 
    
    sentences = [list() for _ in range(L)] #L is the max sentences number
    for j in range(4): #four sentences
        for i in range(n):
            if (l[i] != 0):
                if (l[i]-1 < j):
                    continue
                sentence = nltk.sent_tokenize(small_yelp[i])[j] # several sentences of document i
                sentences[j].append(sentence)  
                
    # 1.2 Clean Doc
    print('Clean Doc')
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    '''
    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split()])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return punc_free #normalized
    '''
    def clean(doc):
        stop_free = " ".join([word for word in doc.lower().split() if word not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


    text_0 = sentences[0] # 9158
    text_1 = sentences[1] # 9003
    text_2 = sentences[2] # 8525
    text_3 = sentences[3] # 7321

    text_0_clean = [clean(doc)for doc in text_0] # text_clean = [clean(doc).split()  for doc in text]
    text_1_clean = [clean(doc)for doc in text_1] # text_clean = [clean(doc).split()  for doc in text]
    text_2_clean = [clean(doc)for doc in text_2] # text_clean = [clean(doc).split()  for doc in text]
    text_3_clean = [clean(doc)for doc in text_3] # text_clean = [clean(doc).split()  for doc in text]
    text_all_clean = [clean(doc)for doc in small_yelp]

    
    return text_all_clean, l, text_0_clean, text_1_clean, text_2_clean, text_3_clean
