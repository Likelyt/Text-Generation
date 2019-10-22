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
from masked_cross_entropy import masked_cross_entropy

SOS_token = 1


def train(input_batches, input_lengths, target_batches, target_lengths, 
          input_topic, target_topic,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, batch_size):

    # Encoder initialization
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
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
        decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, input_topic, encoder_outputs)
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

    # Clip gradient norms
    # ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    # dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
    # Optimization
    encoder_optimizer.step()
    decoder_optimizer.step()
    #lda_topic_opitimizer.step()

    # Return loss
    return torch.tensor(loss.item())#, ec, dc



def lda_topic_loss(input_topic, target_topic, encoder_hidden, lda_topic, batch_size, lda_topic_opitimizer):
    y_hat = lda_topic(input_topic, encoder_hidden)
    log_loss = -torch.sum(torch.mul(torch.log(y_hat), target_topic), dim = 0)
    loss_lda = torch.sum(log_loss)/batch_size
    lda_topic_opitimizer.zero_grad()
    loss_lda.sum().backward()
    print("LDA MLP Topic transition: Log Loss: {:.4f}".format(loss_lda))
    return loss_lda

def ntm_topic_loss(input_topic, target_topic, encoder_hidden, ntm_topic, batch_size, lda_topic_opitimizer):
    y_hat = ntm_topic(input_topic, encoder_hidden)
    print(-torch.sum(torch.log(y_hat), dim = 0)*0.01)
    log_loss = -torch.sum(torch.mul(torch.log(y_hat), target_topic), dim = 0)
    loss_ntm = torch.sum(log_loss)/batch_size
    lda_topic_opitimizer.zero_grad()
    loss_ntm.sum().backward()
   lda_topic_opitimizer.step()
    print("NTM MLP Topic transition: Log Loss: {:.4f}".format(loss_ntm))
    return loss_ntm


def train_mlp(input_batches, input_lengths,
                input_topic, target_topic, 
                encoder, decoder, mlp_topic, topic_choice,
                lda_topic_opitimizer, batch_size):

    lda_topic_opitimizer.zero_grad()   
    # Run words through encoder
    # encoder_outputs: max_input_seq_len x batch_size x hidden_size
    # encoder_hidden: 1 x batch_size x hidden_size
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    if topic_choice == 'LDA':
        loss = lda_topic_loss(input_topic, target_topic, encoder_hidden, mlp_topic, batch_size, lda_topic_opitimizer)
    else:
        loss = ntm_topic_loss(input_topic, target_topic, encoder_hidden, mlp_topic, batch_size, lda_topic_opitimizer)
    sys.stdout.flush()

    return loss

