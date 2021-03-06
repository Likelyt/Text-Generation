#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:39:15 2018

@author: liyt
"""
import numpy as np
import random
import torch

def random_batch(data, input_voc, output_voc, batch_size, USE_CUDA = True, val = False):
    
    PAD_token = 0
    SOS_token = 1 
    EOS_token = 2


    def indexes_from_sentence(vocab_dict, sentence):
        return [vocab_dict.word2index[word] for word in sentence.split(' ')[:50]] + [EOS_token]

    # Pad a with the PAD symbol
    def pad_seq(seq, max_length):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

    # Create Batch for sentence 0 and 1
    group = len(data)// batch_size
    group_reminder = len(data) % batch_size
    input_seqs = [[] for _ in range(group+1)]
    target_seqs = [[] for _ in range(group+1)]
    input_topic = [[] for _ in range(group+1)]
    target_topic = [[] for _ in range(group+1)]

    # All data is been sampled
    sample_id = random.sample(range(len(data)), k = len(data))

    # Select the data:
    for i in range(group):
        sample_x = sample_id[i*batch_size:(i+1)*batch_size]
        for j in range(batch_size):
            pair = data[sample_x[j]]
            target_seqs[i].append(indexes_from_sentence(output_voc, pair[1]))
            target_topic[i].append(pair[3])
            if val == False:
                input_seqs[i].append(indexes_from_sentence(input_voc, pair[0]))
                input_topic[i].append(pair[2])
            else:
                input_seqs[i].append(indexes_from_sentence(input_voc, pair[0]))
                input_topic[i].append(pair[2])

    # For reminder
    """
    sample_x = sample_id[(i+1)*batch_size:]
    for j in range(group_reminder):
        pair = data[sample_x[j]]
        input_seqs[i+1].append(indexes_from_sentence(input_voc, pair[0]))
        target_seqs[i+1].append(indexes_from_sentence(output_voc, pair[1]))
    """



    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = [[] for _ in range(group)]

    input_var = [[] for _ in range(group)]
    input_lengths = [[] for _ in range(group)]
    target_var = [[] for _ in range(group)]
    target_lengths = [[] for _ in range(group)]

    #padded
    input_padded = [[] for _ in range(group)]
    target_padded = [[] for _ in range(group)]

    input_topics = [[] for _ in range(group)]
    target_topics = [[] for _ in range(group)]

    for i in range(group):
        seq_pairs[i] = sorted(zip(input_seqs[i], target_seqs[i], input_topic[i], target_topic[i]), key=lambda p: len(p[0]), reverse=True)
        input_seqs[i], target_seqs[i], input_topics[i], target_topics[i] = zip(*seq_pairs[i])

        # For input and target sequences, get array of lengths and pad with 0s to max length
        input_lengths[i] = [len(s) for s in input_seqs[i]]
        input_padded[i] = [pad_seq(s, max(input_lengths[i])) for s in input_seqs[i]]
        target_lengths[i] = [len(s) for s in target_seqs[i]]
        target_padded[i] = [pad_seq(s, max(target_lengths[i])) for s in target_seqs[i]]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var[i] = torch.LongTensor(input_padded[i]).transpose(0, 1)
        target_var[i] = torch.LongTensor(target_padded[i]).transpose(0, 1)
        input_topics[i] = torch.from_numpy(np.array(input_topics[i])).float()
        target_topics[i] = torch.from_numpy(np.array(target_topics[i])).float()
    

    """
    input_var: seq_len x batch_size
    input_lengths: batch_size

    """
    return input_var, input_lengths, target_var, target_lengths, input_topics, target_topics



def batch_evaluate(x, input_voc, batch_size, USE_CUDA = True):

    PAD_token = 0
    SOS_token = 1 
    EOS_token = 2


    def indexes_from_sentence(vocab_dict, sentence):
        return [vocab_dict.word2index[word] for word in sentence.split(' ')[:50]] + [EOS_token]

    # Pad a with the PAD symbol
    def pad_seq(seq, max_length):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

    # Create Batch for sentence 0 and 1
    input_seqs = []

    # Select the data:
    for i in range(batch_size):
        pair = x[i]
        input_seqs.append(indexes_from_sentence(input_voc, pair))

    # Zip into pairs, sort by length (descending), unzip
    input_var = torch.LongTensor(input_seqs).transpose(0, 1)

    """
    input_var: seq_len x batch_size
    input_lengths: batch_size
    """
    return input_var



def wrapper_data_with_topic(data, topic, choice):
    #1: encoder sentence
    #2: decoder sentence
    #3: encoder topic distribution
    #3: decoder topic distribution
    wrapper = [[] for _ in range(len(data))]
    if choice == 'LDA':
        for i in range(len(data)):
            wrapper[i].append(data[i][0])
            wrapper[i].append(data[i][1])
            wrapper[i].append(topic[i].tolist())
            wrapper[i].append(topic[i+1].tolist())
    else:
        for i in range(len(data)):
            wrapper[i].append(data[i][0])
            wrapper[i].append(data[i][1])
            wrapper[i].append(topic[i])
            wrapper[i].append(topic[i+1])
    return wrapper
       














