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
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 30
PAD_token = 0
SOS_token = 1
EOS_token = 2


def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')[:50]] + [EOS_token]

def perplex(probs, indexes, batch_size):
    ppx_matrix = []
    for i in range(batch_size):
        for j in range(len(indexes[i])):
            if indexes[i][j] == EOS_token:
                ppx_matrix.append(probs[i][:j].cpu().numpy())
                break
        if j == len(indexes[i])-1:
            ppx_matrix.append(probs[i].cpu().numpy())
    ppx = 0
    for i in range(batch_size):
        if ppx_matrix[i].shape[0] == 0:
            continue
        else:
            ppx += sum(np.log(ppx_matrix[i]))/ppx_matrix[i].shape[0]
    ppx = np.exp(-ppx/batch_size)
    return ppx


def perplex_tar(probs, indexes, tar_lengths, batch_size):
    ppx_matrix = []
    for i in range(batch_size):
        if tar_lengths[i] < MAX_LENGTH:
            ppx_matrix.append(probs[i][:tar_lengths[i]].cpu().numpy())
        else:
            ppx_matrix.append(probs[i][:MAX_LENGTH].cpu().numpy())
    ppx = 0
    for i in range(batch_size):
        if ppx_matrix[i].shape[0] == 0:
            continue
        else:
            ppx += sum(np.log(ppx_matrix[i]))/ppx_matrix[i].shape[0]
    ppx = np.exp(-ppx/batch_size)
    return ppx

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

def evaluate(input_seq, input_lengths, tar_seq, tar_lengths, 
    x,
    batch_size, 
    encoder, decoder, model_vae_1, TT,
    input_voc, output_voc, 
    h1_val_samples=2, h2_val_samples = 2, 
    max_length=MAX_LENGTH, USE_CUDA = True):


    # loss
    loss = 0
    loss_1 = 0
    loss_2 = 0
    loss_3 = 0


    if USE_CUDA:
        input_batches = input_seq.cuda()
        x = x.float().cuda()
        
    # Set to not-training mode to disable dropout
    '''
    encoder.train(False)
    decoder.train(False)
    model_vae_1.train(False)
    TT.train(False)
    '''
    encoder.training  = False
    decoder.training = False
    model_vae_1.training = False
    TT.training = False

    # Run through encoder
    # encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)

    # produce the mu, log_var and topic vector
    y_hat, mu, log_var, z = model_vae_1(x, h1_val_samples)




    if h1_val_samples == 1 and h2_val_samples == 1:
        y_hat_1, mu_1, log_var_1, t_1 = model_vae_1(x, h1_val_samples)
        mu_2, log_var_2, t_2 = TT(t_1, encoder_hidden[:decoder.n_layers], h2_val_samples)

        for jj in range(batch_size):
            Norm = torch.distributions.multivariate_normal.MultivariateNormal(mu_2[jj], torch.diag(torch.exp(log_var_2[jj])))
            loss_3 += Norm.log_prob(t_2[jj])

        # Compute loss_1
        # Prepare input and output variables
        decoder_input = torch.LongTensor([SOS_token] * batch_size) # decoder_input: 1 x batch_size
        decoder_hidden = encoder_hidden[:decoder.n_layers] 

        USE_CUDA = True
        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Run through decoder one time step at a time
        # Store output words and attention states
        decoded_words = [[] for i in range(batch_size)]
        decoder_attentions = torch.zeros(batch_size, max_length + 1, 2*max_length)
    
        topv_p_all = torch.zeros([batch_size,1]).cuda()
        topi_p_all = torch.ones([batch_size,1]).cuda()

        # Run through decoder
        max_length = min(tar_seq.size()[0], MAX_LENGTH)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, t_2, encoder_outputs)
            decoder_attentions[:,di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).squeeze(1).cpu().data
            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            topv_p, topi_p = F.softmax(decoder_output.data, dim = 1).topk(1)
            topv_p_all = torch.cat((topv_p_all, topv_p), dim=1)
            topi_p_all = torch.cat((topi_p_all, topi_p.float()), dim=1)
            decoded_word, next_word = greedy_search(topv, topi, batch_size, output_voc)
            for k in range(batch_size):
                decoded_words[k].append(decoded_word[k][0])
            decoder_input = torch.LongTensor(next_word).cuda()

        topv_p_all = topv_p_all[:,1:]
        topi_p_all = topi_p_all[:,1:]

        #ppx = perplex(topv_p_all, topi_p_all, batch_size)


        decoded_words_one_EOS = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(len(decoded_words[i])):
                if decoded_words[i][j] == 'EOS':
                    decoded_words_one_EOS[i] = decoded_words[i][:j+1]
                    break
            if j == len(decoded_words[i])-1:
                decoded_words_one_EOS[i] = decoded_words[i]


    if h1_val_samples == 1 and h2_val_samples > 1:
        y_hat_1, mu_1, log_var_1, t_1 = model_vae_1(x, h1_val_samples)
        mu_2, log_var_2, t_2 = TT(t_1, encoder_hidden[:decoder.n_layers], h2_val_samples)
        # Compute loss_2
        for ii in range(h2_val_samples):
            for jj in range(batch_size):
                Norm = torch.distributions.multivariate_normal.MultivariateNormal(mu_2[jj], torch.diag(torch.exp(log_var_2[jj])))
                loss_3 += Norm.log_prob(t_2[ii][jj])


        # topic_idx represent the ith sampling for t_2
        for topic_idx in range(len(t_2)):  

            # Prepare input and output variables
            decoder_input = torch.LongTensor([SOS_token] * batch_size) # decoder_input: 1 x batch_size
            decoder_hidden = encoder_hidden[:decoder.n_layers]
                
            # Move new Variables to CUDA
            USE_CUDA = True
            if USE_CUDA:
                decoder_input = decoder_input.cuda()    

            # Run through decoder one time step at a time
            # Store output words and attention states
            decoded_words = [[] for i in range(batch_size)]
            decoder_attentions = torch.zeros(batch_size, max_length + 1, 2*max_length)
        
            topv_p_all = torch.zeros([batch_size,1]).cuda()
            topi_p_all = torch.ones([batch_size,1]).cuda()

            max_length = min(tar_seq.size()[0], MAX_LENGTH)
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, t_2[topic_idx], encoder_outputs)
                decoder_attentions[:,di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).squeeze(1).cpu().data
                # Choose top word from output
                topv, topi = decoder_output.data.topk(1)
                topv_p, topi_p = F.softmax(decoder_output.data, dim = 1).topk(1)
                topv_p_all = torch.cat((topv_p_all, topv_p), dim=1)
                topi_p_all = torch.cat((topi_p_all, topi_p.float()), dim=1)
                decoded_word, next_word = greedy_search(topv, topi, batch_size, output_voc)
                for k in range(batch_size):
                    decoded_words[k].append(decoded_word[k][0])
                decoder_input = torch.LongTensor(next_word).cuda()


            topv_p_all = topv_p_all[:,1:]
            topi_p_all = topi_p_all[:,1:]

            decoded_words_one_EOS = [[] for i in range(batch_size)]
            for i in range(batch_size):
                for j in range(len(decoded_words[i])):
                    if decoded_words[i][j] == 'EOS':
                        decoded_words_one_EOS[i] = decoded_words[i][:j+1]
                        break
                if j == len(decoded_words[i])-1:
                    decoded_words_one_EOS[i] = decoded_words[i]




    if h1_val_samples > 1 and h2_val_samples == 1:
        y_hat_1, mu_1, log_var_1, t_1 = model_vae_1(x, h1_val_samples)

        for l_1 in range(h1_val_samples):
            mu_2, log_var_2, t_2 = TT(t_1[l_1], encoder_hidden[:decoder.n_layers], h2_val_samples)

            for jj in range(batch_size):
                Norm = torch.distributions.multivariate_normal.MultivariateNormal(mu_2[jj], torch.diag(torch.exp(log_var_2[jj])))
                loss_3 += Norm.log_prob(t_2[jj])


            # Compute loss_1
            # Prepare input and output variables
            decoder_input = torch.LongTensor([SOS_token] * batch_size) # decoder_input: 1 x batch_size
            decoder_hidden = encoder_hidden[:decoder.n_layers] 

            USE_CUDA = True
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            # Run through decoder one time step at a time
            # Store output words and attention states
            decoded_words = [[] for i in range(batch_size)]
            decoder_attentions = torch.zeros(batch_size, max_length + 1, 2*max_length)
    
            topv_p_all = torch.zeros([batch_size,1]).cuda()
            topi_p_all = torch.ones([batch_size,1]).cuda()

            # Run through decoder
            max_length = min(tar_seq.size()[0], MAX_LENGTH)
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, t_2, encoder_outputs)
                decoder_attentions[:,di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).squeeze(1).cpu().data
                # Choose top word from output
                topv, topi = decoder_output.data.topk(1)
                topv_p, topi_p = F.softmax(decoder_output.data, dim = 1).topk(1)
                topv_p_all = torch.cat((topv_p_all, topv_p), dim=1)
                topi_p_all = torch.cat((topi_p_all, topi_p.float()), dim=1)
                decoded_word, next_word = greedy_search(topv, topi, batch_size, output_voc)
                for k in range(batch_size):
                    decoded_words[k].append(decoded_word[k][0])
                decoder_input = torch.LongTensor(next_word).cuda()


            decoded_words_one_EOS = [[] for i in range(batch_size)]
            for i in range(batch_size):
                for j in range(len(decoded_words[i])):
                    if decoded_words[i][j] == 'EOS':
                        decoded_words_one_EOS[i] = decoded_words[i][:j+1]
                        break
                if j == len(decoded_words[i])-1:
                    decoded_words_one_EOS[i] = decoded_words[i]


    if h1_val_samples > 1 and h2_val_samples > 1:
        #compute t_1
        y_hat_1, mu_1, log_var_1, t_1 = model_vae_1(x, h1_val_samples)

        #at each t_1, sampling t_2
        for l_1 in range(h1_val_samples):
            mu_2, log_var_2, t_2 = TT(t_1[l_1], encoder_hidden[:decoder.n_layers], h2_val_samples)

            for l_2 in range(h2_val_samples):
                for jj in range(batch_size):
                    Norm = torch.distributions.multivariate_normal.MultivariateNormal(mu_2[jj], torch.diag(torch.exp(log_var_2[jj])))
                    loss_3 += Norm.log_prob(t_2[l_2][jj])

                # Compute loss_1 ---- should average t_2
                
                # Prepare input and output variables
                decoder_input = torch.LongTensor([SOS_token] * batch_size) # decoder_input: 1 x batch_size
                decoder_hidden = encoder_hidden[:decoder.n_layers] 

                # Move new Variables to CUDA
                USE_CUDA = True
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()
                # Run through decoder one time step at a time
                # Store output words and attention states
                decoded_words = [[] for i in range(batch_size)]
                decoder_attentions = torch.zeros(batch_size, max_length + 1, 2*max_length)
        
                topv_p_all = torch.zeros([batch_size,1]).cuda()
                topi_p_all = torch.ones([batch_size,1]).cuda()  


                # Run through decoder
                max_length = min(tar_seq.size()[0], MAX_LENGTH)
                for di in range(max_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, t_2[l_2], encoder_outputs)
                    decoder_attentions[:,di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).squeeze(1).cpu().data
                    # Choose top word from output
                    topv, topi = decoder_output.data.topk(1)
                    topv_p, topi_p = F.softmax(decoder_output.data, dim = 1).topk(1)
                    topv_p_all = torch.cat((topv_p_all, topv_p), dim=1)
                    topi_p_all = torch.cat((topi_p_all, topi_p.float()), dim=1)
                    decoded_word, next_word = greedy_search(topv, topi, batch_size, output_voc)
                    for k in range(batch_size):
                        decoded_words[k].append(decoded_word[k][0])
                    decoder_input = torch.LongTensor(next_word).cuda()


                decoded_words_one_EOS = [[] for i in range(batch_size)]
                for i in range(batch_size):
                    for j in range(len(decoded_words[i])):
                        if decoded_words[i][j] == 'EOS':
                            decoded_words_one_EOS[i] = decoded_words[i][:j+1]
                            break
                    if j == len(decoded_words[i])-1:
                        decoded_words_one_EOS[i] = decoded_words[i]





    # Set back to training mode
    '''
    encoder.train(True)
    decoder.train(True)
    model_vae_1.train(True)
    TT.train(True)
    '''

    encoder.training  = True
    decoder.training = True
    model_vae_1.training = True
    TT.training = True

    return decoded_words, decoded_words_one_EOS, decoder_attentions[:di+1, :len(encoder_outputs)]




def evaluate_and_show_attention(val_input, val_input_lengs, val_target, val_target_lengs, 
    x,
    batch_size, encoder, decoder, model_vae_1, TT,
    input_voc, output_voc, h1_val_samples, h2_val_samples):


    output_words, decoded_words_one_EOS, attentions = evaluate(val_input, val_input_lengs, val_target, val_target_lengs, 
        x,
        batch_size, 
        encoder, decoder, model_vae_1, TT,
        input_voc, output_voc,
        h1_val_samples, h2_val_samples)
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


def evaluate_randomly_bleu(val_input, val_input_lengs, val_target, val_target_lengs,
    x,
    batch_size, 
    encoder, decoder, model_vae_1, TT,
    input_voc, output_voc, h1_val_samples, h2_val_samples, target = True):
    if target == True:
        input_sentence, output_sentences, output_sentences_one_EOS, target_sentence = evaluate_and_show_attention(
             val_input, val_input_lengs, val_target, val_target_lengs,
             x,
             batch_size, 
             encoder, decoder, model_vae_1, TT,
             input_voc, output_voc,
             h1_val_samples, h2_val_samples)
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












