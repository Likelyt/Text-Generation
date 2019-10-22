#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:15:30 2018

@author: liyt
"""
import os
import sys
import time
import random
import torch
import time_since as ts
import random_batch as rb
import random_batch_topic as rbt
from evaluate_bleu import evaluate_randomly_bleu
from evaluate_ppx import evaluate_randomly_ppx
from evaluate_bleu import transform_from_index
from train import train
from plot_loss import showPlot
from score import score_calculator
from ntm_generation import NTM_topic_generation



def sentence_from_word(voc, index):
    return ' '.join([voc.index2word[id] for id in index[:50]])



def trainIters(data, val_data, test_data, 
               data_vae_train, data_vae_val, data_vae_test,
               encoder, decoder, model_vae_1, TT,
               h1_samples, h2_samples,
               h1_val_samples, h2_val_samples,
               input_voc, output_voc, 
               criterion, n_epochs, batch_size, n_val, number_of_topic, opt,
               print_every=1, plot_every=1, learning_rate=0.01, USE_CUDA=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    val_ppx = [] #For test use / index
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every
    
    encoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,encoder.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    decoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,decoder.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    model_vae_1_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model_vae_1.parameters()), lr = learning_rate, betas=(0.99, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    TT_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,TT.parameters()), lr = learning_rate, betas=(0.99, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    epoch = 0
    group = len(data)/batch_size * n_epochs


    while epoch < n_epochs:
        epoch += 1
        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = rb.random_batch(data, input_voc, output_voc, batch_size)
        for step in range(1, len(data)//batch_size+1):
            
            # Trans data to GPU model
            if USE_CUDA:
                input_batches[step-1] = input_batches[step-1].cuda()
                target_batches[step-1]= target_batches[step-1].cuda()
                x = torch.tensor(data = data_vae_train[step-1]).cuda().float()

            # Run the train function
            loss = train(
                input_batches[step-1], input_lengths[step-1],
                target_batches[step-1], target_lengths[step-1],
                x, 
                h1_samples, h2_samples,
                encoder, decoder, model_vae_1, TT,
                encoder_optimizer, decoder_optimizer, model_vae_1_optimizer, TT_optimizer,
                criterion, batch_size, number_of_topic
            )


            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss
        
            # Print
            if step % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = 'Epoch: %d / %d, Accumulate Time to End: %s, (Batch: %d / Batches Num: %d, Percent Run: %.2f%%), Loss:%.4f' % (
                    epoch, n_epochs, ts.time_since(start, (step+(epoch-1)*len(data)/batch_size) / group), 
                    step, len(data)/batch_size, ((epoch-1) * (len(data)/batch_size) + step ) / group * 100, print_loss_avg
                )
                print(print_summary)
                sys.stdout.flush()
    
            # Plot
            if step % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            
        # Validation - Part
        print('\nVal Epoch: %d' % (epoch))
        # Batches evaluate
        val_input_batches, val_input_lengths, val_target_batches, val_target_lengths = rb.random_batch(val_data, input_voc, output_voc, batch_size)
        
        val_input = [[] for _ in range(len(val_data)//batch_size)]
        val_output = [[] for _ in range(len(val_data)//batch_size)]
        val_output_one_EOS = [[] for _ in range(len(val_data)//batch_size)]
        val_target = [[] for _ in range(len(val_data)//batch_size)]
        
        ppx = [[] for _ in range(len(val_data)//batch_size)]

        for i in range(len(val_data)//batch_size):
            val_input[i], val_output[i], val_output_one_EOS[i], val_target[i], ppx[i] = evaluate_randomly_ppx(
                 val_input_batches[i], val_input_lengths[i], val_target_batches[i], val_target_lengths[i],
                 data_vae_val[i],
                 batch_size, 
                 encoder, decoder, model_vae_1, TT,
                 input_voc, output_voc,
                 h1_val_samples, h2_val_samples)


        # Print some results
        '''
        for l in range(10):
            print('\nVal Sample of Golden Input: %d' % (l+1))
            print('\nInput Sentence: <', transform_from_index(val_input_batches[i][:,l], output_voc))
            print('Target Sentence: <', transform_from_index(val_target_batches[i][:,l], output_voc))
            print('Output Sentence: <', val_output_one_EOS[i][l])
            sys.stdout.flush()   
        '''
        
        # Calculate Bleu Score    
        val_input = [[] for _ in range(len(val_data)//batch_size)]
        val_output = [[] for _ in range(len(val_data)//batch_size)]
        val_output_one_EOS = [[] for _ in range(len(val_data)//batch_size)]
        val_target = [[] for _ in range(len(val_data)//batch_size)]

        for i in range(len(val_data)//batch_size):
            val_input[i], val_output[i], val_output_one_EOS[i], val_target[i] = evaluate_randomly_bleu(
                 val_input_batches[i], val_input_lengths[i], val_target_batches[i], val_target_lengths[i],
                 data_vae_val[i],
                 batch_size, 
                 encoder, decoder, model_vae_1, TT,
                 input_voc, output_voc,
                 h1_val_samples, h2_val_samples)

        bleu_scores = [[] for t in range(len(val_input))]
        etp_scores = [[] for t in range(len(val_input))]
        div_scores = [[] for t in range(len(val_input))]
        for i in range(len(val_input)):
            temp_index = rb.batch_evaluate(val_output[i], output_voc, batch_size, USE_CUDA = True)
            bleu_scores[i], etp_scores[i], div_scores[i] = score_calculator(temp_index, val_target[i]) 

        etp_score = [sum(col) / float(len(col)) for col in zip(*etp_scores)]
        div_socre = [sum(col) / float(len(col)) for col in zip(*div_scores)]
        blue_score = [sum(col) / float(len(col)) for col in zip(*bleu_scores)]
        print('\nOutput as Input\netp score: %.4f, %.4f, %.4f, %.4f'% (etp_score[0], etp_score[1], etp_score[2], etp_score[3]))
        print('Div score: %.4f, %.4f, %.4f, %.4f'% (div_socre[0], div_socre[1], div_socre[2], div_socre[3]))
        print('Bleu score: %.4f, %.4f, %.4f, %.4f' % (blue_score[0], blue_score[1], blue_score[2], blue_score[3]))
        print('Perplexity of Golden Input:  %.4f' % (sum(ppx)/len(ppx)))
        sys.stdout.flush()        


        #Save perplexity/loss and then use for testing
        val_ppx.append(sum(ppx)/len(ppx))

        # Check points
        checkpoint = {
            'encoder_model': encoder,
            'decoder_model': decoder,
            'vae_model':model_vae_1,
            'topic_transition_model': TT,
            'input_voc': input_voc, 
            'output_voc': output_voc,
            'encoder_hidden_size': encoder.hidden_size,
            'decoder_hidden_size': decoder.hidden_size,
            'epoch': epoch,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'loss': loss,
            'perplexity': sum(ppx)/len(ppx)
        }

        val_model_name = '%s%s_%d%s%d%s%d%s%d%s%d%s%d%s%d%s%d' % (
            'model_result/', opt.review_name, len(test_data),
            '_hidden_size_', opt.hidden_size,
            '_E_layer_', encoder.n_layers,
            '_D_layer_', decoder.n_layers,
            '_E_bi_',encoder.bi,
            '_T_',opt.n_topic,
            '_L1_',opt.h1_testing_samples,
            '_L2_',opt.h2_testing_samples
            )
        val_model_epoch = '%s%s%d%s' %(
            val_model_name, '_epoch_', epoch, '.pt'
            )

        torch.save(checkpoint, val_model_epoch)
        print("Save model as %s" % val_model_epoch)

        
    showPlot(plot_losses)
    return val_ppx.index(min(val_ppx))+1, val_model_name


























