#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:36:36 2018

@author: liyt
"""
import sys
import random
import torch
import random_batch as rb

MAX_LENGTH = 30
PAD_token = 0
SOS_token = 1
EOS_token = 2


def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')[:50]] + [EOS_token]



def greedy_search(topv, topi, batch_size, output_voc):
    decoded_words = [[] for _ in range(batch_size)]
    next_word = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        ni = topi[i][0]
        next_word[i] = torch.LongTensor([ni])
        if ni == 2:
            decoded_words[i].append('EOS')
        else:
            decoded_words[i].append(output_voc.index2word[ni.item()])
    return decoded_words,  next_word

def evaluate(input_seq, input_lengths, val_input_topic, val_target_topic, 
    batch_size, encoder, decoder, input_voc, output_voc, max_length=MAX_LENGTH, USE_CUDA = True):
    '''
    input_seqs = indexes_from_sentence(input_voc, input_seq)
    input_lengths = torch.tensor([len(input_seqs)])
    input_batches = torch.LongTensor(input_seqs).view(-1, 1)
    '''

    #input_seq = torch.tensor(input_seq)

    if USE_CUDA:
        input_batches = input_seq.cuda()
        val_target_topic = val_target_topic.cuda()

        
    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([SOS_token] * batch_size) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    
    if USE_CUDA:
        decoder_input = decoder_input.cuda()


    # Store output words and attention states
    decoded_words = [[] for i in range(batch_size)]
    decoder_attentions = torch.zeros(batch_size, max_length + 1, 2*max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, val_target_topic, encoder_outputs)
        decoder_attentions[:,di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).squeeze(1).cpu().data
        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        decoded_word, next_word = greedy_search(topv, topi, batch_size, output_voc)
        for k in range(batch_size):
            decoded_words[k].append(decoded_word[k][0])
        decoder_input = torch.LongTensor(next_word).cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    decoded_words_one_EOS = [[] for i in range(batch_size)]
    for i in range(batch_size):
        for j in range(len(decoded_words[i])):
            if decoded_words[i][j] == 'EOS':
                decoded_words_one_EOS[i] = decoded_words[i][:j+1]
                break
    return decoded_words, decoded_words_one_EOS, decoder_attentions[:di+1, :len(encoder_outputs)]



def evaluate_and_show_attention(val_input, val_input_lengs, val_target, val_target_lengs, val_input_topic, val_target_topic,
    batch_size, encoder, decoder, input_voc, output_voc):


    output_words, decoded_words_one_EOS, attentions = evaluate(val_input, val_input_lengs, val_input_topic, val_target_topic,
        batch_size, encoder, decoder, input_voc, output_voc)
    output_sentences = [[] for _ in range(batch_size)]
    output_sentences_one_EOS = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        output_sentences[i] = ' '.join(output_words[i])
        output_sentences_one_EOS[i] = ' '.join(decoded_words_one_EOS[i])

    '''
    print('Input Sentence: >', input_sentence)
    if target_sentence is not None:
        print('Target Sentence :=', target_sentence)
    print('Output Sentence: <', output_sentence)
    print('Output Sentence length: ',len(output_sentence.split(' ')))
    sys.stdout.flush()
    '''

    return val_input, output_sentences, output_sentences_one_EOS, val_target


def evaluate_randomly(val_input, val_input_lengs, val_target, val_target_lengs, val_input_topic, val_target_topic,
    batch_size, encoder, decoder, input_voc, output_voc, target = True):
    if target == True:
        input_sentence, output_sentences, output_sentences_one_EOS, target_sentence = evaluate_and_show_attention(val_input, val_input_lengs, val_target, val_target_lengs,
            val_input_topic, val_target_topic, batch_size, encoder, decoder, input_voc, output_voc)
        return input_sentence, output_sentences, output_sentences_one_EOS, target_sentence
    else:
        input_sentence, target_sentence = data[0], data[1]
        output_sentence, target_sentence = evaluate_and_show_attention(input_sentence, encoder, decoder, input_voc, output_voc)
        return input_sentence, output_sentence, target_sentence
    

def sentence_from_indexes(output_voc, indexes):
    next_wd = []
    for index in indexes:
        if index == 2:
            break
        else:
            next_wd.append(output_voc.index2word[index.item()])
    return next_wd
    

def transform_from_index(data, output_voc):
    sentence = []
    sentence.append(' '.join(sentence_from_indexes(output_voc, data)))
    return sentence












