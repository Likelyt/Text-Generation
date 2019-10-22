#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:33:02 2018

@author: liyt
"""

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {'PAD':10, 'SOS':10, 'EOS':10, '<Unknown>':10}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "<Unknown>"}
        self.n_words = 4 # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Substitute words below a certain count threshold with unknown 
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v > min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {'PAD':0, 'SOS':1, 'EOS':2, '<Unknown>':3}
        self.word2count = {'PAD':10, 'SOS':10, 'EOS':10, '<Unknown>':10}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "<Unknown>"}
        self.n_words = 4 # Count default tokens

        for word in keep_words:
            self.index_word(word)




def filter_pairs(pairs, sentence_num = 3, Min_Length = 3, Max_Length = 150):
    # Filter Sentence Length
    filtered_pairs = []
    i = 0
    keep_index = []
    for pair in pairs:
        # Two condition must satisfied
        if (len(pair[0].split(' ')) >= Min_Length and len(pair[0].split(' ')) <= Max_Length) \
            and (len(pair[1].split(' ')) >= Min_Length and len(pair[1].split(' ')) <= Max_Length):
                filtered_pairs.append(pair)
                keep_index.append(i)
        i += 1
    return filtered_pairs, keep_index

def prepare_data(voc_input, voc_output, data):
    print("Read %d sentence pairs" % len(data))
    
    data_min_max_length, keep_index = filter_pairs(data)
    print("Filtered to %d pairs" % len(data_min_max_length))
    
    print("Indexing words...")
    for pair in data_min_max_length:
        voc_input.index_words(pair[0])
        voc_input.index_words(pair[1])
        voc_output.index_words(pair[0])
        voc_output.index_words(pair[1])
    
    print('Indexed %d words in input language, %d words in output' % (voc_input.n_words, voc_input.n_words))
    
    return voc_input, voc_input, data_min_max_length, keep_index


# Build vocabulay:
def build_voc(data, Min_count = 3):
    voc_encoder = Voc('Text_generation_encoder')
    voc_decoder = Voc('Text_generation_decoder')
        
    # trim those sentence length less than 2, large than 150
    voc_input, voc_output, data_clean_0, keep_index = prepare_data(voc_encoder, voc_decoder, data)
    
    # Filtering vocabularies -  Vocabulary get small - NOT considering at now
    voc_input.trim(Min_count)  
    voc_output.trim(Min_count)
    
    keep_pairs = [] 
    for pair in data_clean_0:
        input_sentence_0 = pair[0].split(' ')
        output_sentence_1 = pair[1].split(' ')
        
        # Use unknown token substitute the low freq word
        for i in range(len(input_sentence_0)):
            if input_sentence_0[i] not in voc_input.word2index:
                input_sentence_0[i] = '<Unknown>'
        for j in range(len(output_sentence_1)):
            if output_sentence_1[j] not in voc_input.word2index:
                output_sentence_1[j] = '<Unknown>'
        
        input_sentence_0 = ' '.join(word for word in input_sentence_0)
        output_sentence_1 = ' '.join(word for word in output_sentence_1)
        pair = [input_sentence_0, output_sentence_1]
        keep_pairs.append(pair)
      
    #print("Trimmed from %d pairs to %d, %.4f of total" % (len(data_clean_0), len(keep_pairs), len(keep_pairs) / len(data_clean_0)))
    train_encoder = []
    train_decoder = []
    for i in range(len(keep_pairs)):
        train_encoder.append(keep_pairs[i][0])
        train_decoder.append(keep_pairs[i][1])
    # Add <Unknown> to vocabulary.
    voc_input.index_words('<Unknown>')
    voc_output.index_words('<Unknown>')
    
    return keep_pairs, keep_index, train_encoder, train_decoder, voc_input, voc_input # the third input is output

    
    
    
    
    