#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import create_voc_4_raw as create_voc_4

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer


def MaxLength(text):
    # Suppose input data is list of sentence
    max_length = 0
    for i in range(len(text)):
        if max_length < len(text[i].split()):
            max_length = len(text[i].split())
    return max_length

def AvgLength(text):
    # Suppose the input data is pair format:
    avg_length_encode = 0
    avg_length_decode = 0
    for i in range(len(text)):
        avg_length_encode += len(text[i][0].split())
        avg_length_decode += len(text[i][1].split())
    return avg_length_encode/len(text), avg_length_decode/len(text)


def Remove_duplicate(text):
    n = len(text)
    voc_size = len(text[0])
    # Keep first sentence and last sentence
    text_clean = torch.Tensor(int(n/2+1),voc_size)
    text_clean[0] = text[0]
    for i in range(1, n, 2):
        text_clean[int((i-1)/2+1)] = text[i]
    return text_clean



class VAE(nn.Module):
    def __init__(self, input_size, h_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim,input_size)
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc21(h)
        log_var = self.fc22(h)
        return mu, log_var
    def parameterize(self, mu, log_var, h1_samples):
        std = torch.exp(log_var/2)
        if h1_samples == 1:
            epsilon = torch.randn_like(std)
            h = mu + epsilon * std 
        else:
            epsilon = torch.randn((h1_samples*std.size(0), std.size(1))).cuda().float()
            epsilon_list = torch.split(epsilon, std.size(0), dim=0) # it returns a tuple
            h = []
            for i in range(h1_samples):
                h.append(mu + epsilon_list[i] * std)
        return h
    def decode(self, h, h1_samples):
        if h1_samples == 1:
            z = F.softmax(h, dim = 1)
            y = F.softmax(self.fc3(z), dim = 1)
        else:
            z = []
            y = []
            for i in range(h1_samples):
                z.append(F.softmax(h[i], dim = 1))
                y.append(F.softmax(self.fc3(z[i]), dim = 1))
        return z, y 
    def forward(self, x, h1_samples):
        mu, log_var = self.encode(x)
        h = self.parameterize(mu, log_var, h1_samples)
        z, y = self.decode(h, h1_samples)
        return y, mu, log_var, z




def NTM_topic_generation(data_name, z_dim, batch_size):
    # 0. Load Data Pairs
    data_pair = np.loadtxt('yelp-data/'+ data_name, delimiter=',', dtype = np.unicode_, encoding='utf8').tolist()
    data, keep_index, train_encoder, train_decoder, voc_input, voc_output = create_voc_4.build_voc(data_pair)
    print('There are %d pairs in total' % (len(data)))

    # 1. Transform data to one-hot 
    T = Tokenizer()
    # Odd number is input sentence, even number is target sentence
    data_one_column = [] 
    for i in range(len(data)):
        data_one_column.append(data[i][0])
    data_one_column.append(data[i][1]) # all sentence 22411

    T.fit_on_texts(data_one_column)
    one_hot_docs = T.texts_to_matrix(data_one_column, mode = 'count') 
    print('There are %d words in total' % (len(one_hot_docs[0])))
    vocab_size = len(T.word_index) + 1 # 5441

    encoded_docs = T.texts_to_sequences(data_one_column)
    max_length = MaxLength(data_one_column) # 111
    encoder_avg_len, decoder_avg_len = AvgLength(data_pair)
    print('The max length of sentence is: %d' % (max_length))
    print('The encoder sentences average length is: %.2f\nThe decoder sentences average length is: %.2f' % (encoder_avg_len, decoder_avg_len))

    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


    embeddings_index = dict()
    f = open('Glove/glove.6B.300d.txt', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()

    # create a weight matrix for words in training docs
    Glove_dim = 300
    embedding_matrix = np.zeros((vocab_size, Glove_dim))  #(5441, 100)
    for word, i in T.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    e = Embedding(vocab_size, Glove_dim, weights=[embedding_matrix], input_length=max_length, trainable=False) # 100 is output dimension


    input_size = vocab_size # 5411 

    shf = False
    torch_onehot_docs = torch.from_numpy(one_hot_docs) # 22411, 5441
    #Remove duplicates
    #torch_onehot_docs = Remove_duplicate(torch_onehot_doc)
    data_size = len(torch_onehot_docs)
    train_batches = utils.create_batch(data_size, batch_size, shuffle=shf)
    data_loader = utils.fetch_data(torch_onehot_docs, train_batches)

    data_vae_list=[]
    for i,j in data_loader.items():
        data_vae_list.append(j)
    
    return data_vae_list, input_size








