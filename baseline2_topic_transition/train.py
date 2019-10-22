#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:14:59 2018

@author: liyt
"""
import torch
import time
import os
import sys
import torch.nn as nn
from masked_cross_entropy import masked_cross_entropy

SOS_token = 1


def train(input_batches, input_lengths, target_batches, target_lengths, 
          input_topic, target_topic,
          encoder, decoder, TT,
          encoder_optimizer, decoder_optimizer,
          criterion, batch_size):


    # loss
    loss = 0
    
    # Run words through encoder
    # encoder_outputs: max_input_seq_len x batch_size x hidden_size
    # encoder_hidden: 1 x batch_size x hidden_size
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = torch.LongTensor([SOS_token] * batch_size) # decoder_input: 1 x batch_size

    # Use last (forward) hidden state from encoder
    # decoder_hidden : 1 x batch_size x hidden_size
    decoder_hidden = encoder_hidden[:decoder.n_layers] 
    
    max_target_length = max(target_lengths)
    # all_decoder_outputs: max_target_length x batch_size x output_vocab_size
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)
    
    # Move new Variables to CUDA
    USE_CUDA = True
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
    
    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, target_topic, encoder_outputs)
        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq, torch.Size([batch, seq, vocab_size])
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq, torch.Size([batch, seq, vocab_size])
        target_lengths # Batch
    )


    # 2. Calculate all the loss
    loss.backward()
    
    # Optimization
    encoder_optimizer.step()
    decoder_optimizer.step()
    #lda_topic_opitimizer.step()

    # Return loss
    return torch.tensor(loss.item())#, ec, dc


'''
def lda_topic_loss(input_topic, target_topic, encoder_hidden, lda_topic, batch_size, lda_topic_opitimizer):
    y_hat = lda_topic(input_topic, encoder_hidden)
    log_loss = -torch.sum(torch.mul(torch.log(y_hat), target_topic), dim = 0)
    loss_lda = torch.sum(log_loss)/batch_size
    lda_topic_opitimizer.zero_grad()
    loss_lda.sum().backward()
    print("LDA MLP Topic transition: Log Loss: {:.4f}".format(loss_lda))
    return loss_lda

'''

def ntm_topic_loss(input_topic, target_topic, encoder_hidden, TT, batch_size, TT_optimizer):
    TT_optimizer.zero_grad()
    y_hat = TT(input_topic, encoder_hidden)
    #print(input_topic[0])
    #print(target_topic[0])
    #print(y_hat[0])
    
    #mse_loss = nn.MSELoss()
    #log_loss = mse_loss(y_hat, target_topic)
    log_loss = -torch.sum(torch.mul(torch.log(y_hat), target_topic), dim = 1)
    loss_ntm = torch.sum(log_loss)/batch_size

    loss_ntm.sum().backward()
    TT_optimizer.step()

    print("NTM MLP Topic transition: Log Loss: {:.4f}".format(loss_ntm))
    return loss_ntm


def train_mlp(input_batches, input_lengths, input_topic, target_topic, 
                encoder, decoder, TT,
                topic_choice,
                TT_optimizer, batch_size):

    TT_optimizer.zero_grad()   
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    '''
    if topic_choice == 'LDA':
        loss = lda_topic_loss(input_topic, target_topic, encoder_hidden, mlp_topic, batch_size, lda_topic_opitimizer)
    '''
    if topic_choice == 'NTM':
        loss = ntm_topic_loss(input_topic, target_topic, encoder_hidden[:decoder.n_layers], TT, batch_size, TT_optimizer)

    return loss

