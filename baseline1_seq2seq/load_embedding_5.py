# Load embedding matrix with dimension 300

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import create_voc_4 as create_voc_4

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer



EOS_token = 2
def indexes_from_sentence(vocab_dict, sentence):
    return [vocab_dict.word2index[word] for word in sentence.split(' ')[:50]] + [EOS_token]

def load_embedding(data, voc_input):

	data_one_column = [] 
	for i in range(len(data)):
		data_one_column.append(data[i][0])

	data_one_column.append(data[i][1]) # all sentence 22411
	Token_num = 0
	for i in data_one_column:
		Token_num += len(i)

	print('There are %d tokens in data' % (Token_num))

	index_docs = []
	for i in data_one_column:
		index_docs.append(indexes_from_sentence(voc_input, i))

	vocab_size = 0
	for i in range(len(index_docs)):
		if vocab_size <= max(index_docs[i]):
			vocab_size = max(index_docs[i])
        
	vocab_size = vocab_size + 1


	embeddings_index = dict()
	f = open('Glove/glove.6B.300d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs

	f.close()

	# create a weight matrix for words in training docs
	Glove_dim = 300
	embedding_matrix = np.zeros((vocab_size, Glove_dim))  #(5454, 100)
	for word, i in voc_input.word2index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	embedding_matrix = torch.from_numpy(embedding_matrix).float()
	return embedding_matrix






