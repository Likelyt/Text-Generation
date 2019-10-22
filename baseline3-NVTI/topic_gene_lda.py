#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:20:19 2018

@author: liyt
"""
import json
import os
import lda
import numpy as np
import string
import utils as utils
from sklearn.feature_extraction.text import CountVectorizer


def LDA_topic(topic_num):

    # 0. Load Data Pairs
    # load keep index for 24K
    keep_index = np.loadtxt('yelp-data/keep_index_24K.txt', delimiter=',', dtype = np.int)
    data_raw = np.loadtxt('yelp-data/data_pair_24K_topic_raw.txt', delimiter=',', dtype = np.str).tolist()
    data = []
    for i in keep_index:
        data.append(data_raw[i])
    print('There are %d pairs in total' % (len(data)))


    # Odd number is input sentence, even number is target sentence
    data_one_column = [] 
    data_one_column.append(data[0][0])
    for i in range(len(data)):
        data_one_column.append(data[i][1])

    vectorizer = CountVectorizer()
    vectorizer.fit(data_one_column)

    vector_all = vectorizer.transform(data_one_column) 
    n = vector_all.shape[0]
    vocab_size = vector_all.shape[1]
    print('There are %d sentences, %d distinct words' % (n, vocab_size)) 

    #2. lda
    n_topics = topic_num
    n_iter = 1000

    model = lda.LDA(n_topics, n_iter, random_state=1)
    X = vector_all.toarray()
    model.fit(X)
    topic_word = model.topic_word_  
    n_top_words = 20

    for i, topic_dist in enumerate(topic_word):
        #print(i)
        topic_words = list()
        tt = np.argsort(topic_dist)[:-(n_top_words+1):-1]
        corr_prob = np.sort(topic_dist)[:-(n_top_words+1):-1]
        for j in range(n_top_words):
            topic_words.append(list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(tt[j])])
        print('Topic {}: {}'.format(i+1, ' '.join(topic_words)))
        #print('Topic {}: {}'.format(i, ', '.join(str(f) for f in np.round(corr_prob,3))))
        #print('Topic {}: {}\n'.format(i, '+'.join(str(f) for f in zip(np.round(corr_prob,3), topic_words))))

    # Beta matrix: topic number, vocabulary number
    print('Beta matrix:\ntopic numbers: %d\n' % (topic_word.shape[0]))
    #np.savetxt('latent_topic_distribution/LDA_beta_matrix_24K_T_5.txt', topic_word, delimiter=',')

    # Sentence topic distribution
    print('Topic distribution:\ntopic numbers: %d' % (topic_word.shape[0]))
    doc_topic = model.doc_topic_
    #np.savetxt('latent_topic_distribution/LDA_sentence_topic_distribution_24K_T_5.txt', doc_topic, delimiter=',')

    return doc_topic, topic_word


















    



















