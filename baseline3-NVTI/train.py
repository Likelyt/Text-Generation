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
import math
from masked_cross_entropy import masked_cross_entropy
from MLP_topic_transition import Topic_Transition

SOS_token = 1


def train(input_batches, input_lengths, target_batches, target_lengths, 
          x, 
          h1_samples, h2_samples,
          encoder, decoder, model_vae_1, TT,
          encoder_optimizer, decoder_optimizer, model_vae_1_optimizer, TT_optimizer,
          criterion, batch_size, J):
    # J: Number of topic

    # Encoder initialization
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    model_vae_1_optimizer.zero_grad()
    TT_optimizer.zero_grad()

    # loss
    loss = 0
    loss_1 = 0
    loss_2 = 0
    loss_3 = 0
    
    
    # Encoder:
    # Run words through encoder
    # encoder_outputs: max_input_seq_len x batch_size x hidden_size
    # encoder_hidden: 1 x batch_size x hidden_size
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # produce the mu, log_var and topic vector
    y_hat, mu, log_var, z = model_vae_1(x, h1_samples)

    if h1_samples == 1:
        y_hat_1, mu_1, log_var_1, t_1 = model_vae_1(x, h1_samples)
        vae_1_log_doc_loss = -torch.sum(torch.mul(torch.log(y_hat_1), x), dim = 1)
        mu_2, log_var_2, t_2 = TT(t_1, encoder_hidden[:decoder.n_layers], h2_samples)

        # h1_samples = 1, h2_samples = 1
        if h2_samples == 1:
            # Compute loss_2
            for jj in range(batch_size):
                Norm = torch.distributions.multivariate_normal.MultivariateNormal(mu_2[jj], torch.diag(torch.exp(log_var_2[jj])))
                loss_3 += Norm.log_prob(t_2[jj])

            # Compute loss_1
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
                decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, t_2, encoder_outputs)
                all_decoder_outputs[t] = decoder_output
                decoder_input = target_batches[t] # Next input is current target
            # Loss calculation and backpropagation
            loss_1 += masked_cross_entropy(
                all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq, torch.Size([batch, seq, vocab_size])
                target_batches.transpose(0, 1).contiguous(), # -> batch x seq, torch.Size([batch, seq, vocab_size])
                target_lengths # Batch
            )


        # h1_samples = 1, h2_samples > 1
        else:
            # Compute loss_2
            for ii in range(h2_samples):
                for jj in range(batch_size):
                    Norm = torch.distributions.multivariate_normal.MultivariateNormal(mu_2[jj], torch.diag(torch.exp(log_var_2[jj])))
                    loss_3 += Norm.log_prob(t_2[ii][jj])


            # Compute loss_1
            for j in range(h2_samples):
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
                    decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, t_2[j], encoder_outputs)
                    all_decoder_outputs[t] = decoder_output
                    decoder_input = target_batches[t] # Next input is current target
                # Loss calculation and backpropagation
                loss_1 += masked_cross_entropy(
                    all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq, torch.Size([batch, seq, vocab_size])
                    target_batches.transpose(0, 1).contiguous(), # -> batch x seq, torch.Size([batch, seq, vocab_size])
                    target_lengths # Batch
                )


        # Compute loss            
        # loss_2 
        for i in range(batch_size):
            for j in range(J):
                loss_2 += 2 + log_var_1[i][j]+ log_var_2[i][j] - mu_1[i][j].pow(2) - torch.exp(log_var_1[i][j])


        kl_div = - ((J/2)*math.log(2*math.pi) + 0.5*loss_2/batch_size + loss_3/(h1_samples*h2_samples*batch_size))
        

        #print(t_2[1])
        #print(mu_1[1].pow(2))
        #print(torch.exp(log_var_1[1]))
        #print(mu_2[1].pow(2))
        #print(torch.exp(log_var_2[1]))
        #print('%d, %d'% (h1_samples, h2_samples))
        #print('Seq2Seq Loss: %.2f' % (loss_1/(h1_samples*h2_samples)))
        #print('Analytic Loss: %.2f' % ((J/2)*math.log(2*math.pi) + 0.5*loss_2/batch_size))
        #print('VAE Transition loss: %.2f' % (loss_3.item()/(h1_samples*h2_samples*batch_size)))
        #print('kl_div: %.2f\n' % (kl_div))


    else:
        y_hat_1, mu_1, log_var_1, t_1 = model_vae_1(x, h1_samples)
        for i in range(h1_samples):
            vae_1_log_doc_loss += -torch.sum(torch.mul(torch.log(y_hat_1[i]), x), dim = 1)
            # t_1[i] is the topic distribution for topic 1
            # then compute the topic transition 
            #TT = Topic_Transition(encoder_hidden.size()[2]+ t_1[0].size()[1], 100, t_1[0].size()[1]).cuda()
            # (t_1[i],  encoder_hidden)
            # t_1[i]: batch_size * topic_dimension, 
            # encoder_hidden: batch_size * hidden_size
            mu_2, log_var_2, t_2 = TT(t_1[i], encoder_hidden[:decoder.n_layers], h2_samples)
            # Compute Loss transition loss

            

            for ii in range(h2_samples):
                for jj in range(batch_size):
                    Norm = torch.distributions.multivariate_normal.MultivariateNormal(mu_2[jj], torch.diag(torch.exp(log_var_2[jj])))
                    loss_3 += Norm.log_prob(t_2[ii][jj])

            for j in range(h2_samples):
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
                    decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, t_2[j], encoder_outputs)
                    all_decoder_outputs[t] = decoder_output
                    decoder_input = target_batches[t] # Next input is current target
            
                # Loss calculation and backpropagation
                loss_1 += masked_cross_entropy(
                    all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq, torch.Size([batch, seq, vocab_size])
                    target_batches.transpose(0, 1).contiguous(), # -> batch x seq, torch.Size([batch, seq, vocab_size])
                    target_lengths # Batch
                )


        # Compute loss            
        for i in range(batch_size):
            for j in range(J):
                loss_2 += 2 + log_var_1[i][j]+ log_var_2[i][j] - mu_1[i][j].pow(2) - torch.exp(log_var_1[i][j])

        kl_div = - ((J/2)*math.log(2*math.pi) + 0.5*loss_2/batch_size + loss_3/(h1_samples*h2_samples*batch_size))
        
        #print(mu_2[i].pow(2))
        #print(torch.exp(log_var_2[i]))
        #print('%d, %d'% (h1_samples, h2_samples))
        #print('Seq2Seq Loss: %.2f' % (loss_1/(h1_samples*h2_samples)))
        #print('Fixed Loss: %.2f' % ((J/2)*math.log(2*math.pi) + 0.5*loss_2/batch_size))
        #print('Transition loss: %.2f' % (loss_3.item()/(h1_samples*h2_samples*batch_size)))
        #print('kl_div: %.2f\n' % (kl_div))



    # 2. Calculate all the loss
    loss = loss_1/(h1_samples*h2_samples) + kl_div
    loss.backward()

    # Clip gradient norms
    # ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    # dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
    # Optimization
    encoder_optimizer.step()
    decoder_optimizer.step()
    model_vae_1_optimizer.step()
    TT_optimizer.step()


    # Return loss
    return torch.tensor(loss.item())#, ec, dc




