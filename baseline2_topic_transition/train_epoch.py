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
from evaluate import evaluate_randomly
from evaluate import transform_from_index
from train import train
from plot_loss import showPlot
from score import score_calculator
from train import train_mlp




def sentence_from_word(voc, index):
    return ' '.join([voc.index2word[id] for id in index[:50]])





def trainIters(data, val_data, encoder, decoder, mlp_topic, topic_choice,
                input_voc, output_voc, 
                criterion, n_epochs, batch_size, n_val, mlp_epoch,
                print_every=1, plot_every=1, learning_rate=0.01, USE_CUDA=True):
    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    lda_topic_opitimizer = torch.optim.Adam(mlp_topic.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    epoch = 0
    group = len(data)/batch_size * n_epochs



    while epoch < n_epochs:
        epoch += 1
        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths, input_t, target_t = rbt.random_batch(data, input_voc, output_voc, batch_size)

        for step in range(1, len(data)//batch_size+1):

            # Trans data to GPU mode
            if USE_CUDA:
                input_batches[step-1] = input_batches[step-1].cuda()
                target_batches[step-1]= target_batches[step-1].cuda()
                input_t[step-1] = input_t[step-1].cuda()
                target_t[step-1] = target_t[step-1].cuda()
            # Run the train function
            loss = train(
                    input_batches[step-1], input_lengths[step-1],
                    target_batches[step-1], target_lengths[step-1],
                    input_t[step-1], target_t[step-1],
                    encoder, decoder, mlp_topic,
                    encoder_optimizer, decoder_optimizer, lda_topic_opitimizer, criterion, batch_size
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
        val_input_batches, val_input_lengths, val_target_batches, val_target_lengths, val_input_t, val_target_t = rbt.random_batch(val_data, input_voc, output_voc, batch_size)
        val_input = [[] for _ in range(len(val_data)//batch_size)]
        val_output = [[] for _ in range(len(val_data)//batch_size)]
        val_output_one_EOS = [[] for _ in range(len(val_data)//batch_size)]
        val_target = [[] for _ in range(len(val_data)//batch_size)]

        for i in range(len(val_data)//batch_size):
            val_input[i], val_output[i], val_output_one_EOS[i], val_target[i] = evaluate_randomly(val_input_batches[i], val_input_lengths[i], val_target_batches[i], val_target_lengths[i],
                 val_input_t[i], val_target_t[i], batch_size, encoder, decoder, input_voc, output_voc)


        # Print some results
        for l in range(10):
            print('\nVal Sample: %d' % (l+1))
            print('\nInput Sentence: <', transform_from_index(val_input_batches[i][:,l], output_voc))
            print('Target Sentence: <', transform_from_index(val_target_batches[i][:,l], output_voc))
            print('Output Sentence: <', val_output_one_EOS[i][l])
            sys.stdout.flush()   

        # Calculate Bleu Score    
        bleu_scores = [[] for t in range(len(val_input))]
        etp_scores = [[] for t in range(len(val_input))]
        div_scores = [[] for t in range(len(val_input))]
        for i in range(len(val_input)):
            temp_index = rb.batch_evaluate(val_output[i], output_voc, batch_size, USE_CUDA = True)
            bleu_scores[i], etp_scores[i], div_scores[i] = score_calculator(temp_index, val_target[i]) 

        etp_score = [sum(col) / float(len(col)) for col in zip(*etp_scores)]
        div_socre = [sum(col) / float(len(col)) for col in zip(*div_scores)]
        blue_score = [sum(col) / float(len(col)) for col in zip(*bleu_scores)]
        print('\netp score: %.4f, %.4f, %.4f, %.4f'% (etp_score[0], etp_score[1], etp_score[2], etp_score[3]))
        print('Div score: %.4f, %.4f, %.4f, %.4f'% (div_socre[0], div_socre[1], div_socre[2], div_socre[3]))
        print('Bleu score: %.4f, %.4f, %.4f, %.4f' % (blue_score[0], blue_score[1], blue_score[2], blue_score[3]))
        sys.stdout.flush()        

        # Check points
        checkpoint = {
            #'model': self.model,
            #'dicts': self.dicts,
            #'opt': self.opts,
            'epoch': epoch,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            #'encoder parameters': encoder.parameters().data,
            #'decoder parameters': decoder.parameters().data,
            'loss': loss,
            #'optim': self.optim,
        }
        model_name = os.path.join('model_result/', "Seq2seq_epoch_%d.pt" % epoch)
        torch.save(checkpoint, model_name)
        print("Save model as %s" % model_name)

    showPlot(plot_losses)


    #compute topic transition with MLP
    #Finish RNN, then train with topic transition
    if topic_choice == 'LDA':
        epoch = 0
        while epoch < mlp_epoch:
            epoch += 1
            # Get training data for this cycle
            input_batches, input_lengths, target_batches, target_lengths, input_t, target_t = rbt.random_batch(data, input_voc, output_voc, batch_size)
            print('\n MLP epoch %d: \n' % (epoch))
            for step in range(1, len(data)//batch_size+1):
                # Trans data to GPU mode
                if USE_CUDA:
                    input_batches[step-1] = input_batches[step-1].cuda()
                    target_batches[step-1]= target_batches[step-1].cuda()
                    input_t[step-1] = input_t[step-1].cuda()
                    target_t[step-1] = target_t[step-1].cuda()
                # Run the train function
                loss_lda = train_mlp(
                        input_batches[step-1], input_lengths[step-1],
                        input_t[step-1], target_t[step-1],
                        encoder, decoder, mlp_topic, topic_choice,
                        lda_topic_opitimizer, batch_size
                )
    else:
        epoch = 0
        while epoch < mlp_epoch:
            epoch += 1
            # Get training data for this cycle
            input_batches, input_lengths, target_batches, target_lengths, input_t, target_t = rbt.random_batch(data, input_voc, output_voc, batch_size)
            print('\n MLP epoch %d: \n' % (epoch))
            for step in range(1, len(data)//batch_size+1):
                # Trans data to GPU mode
                if USE_CUDA:
                    input_batches[step-1] = input_batches[step-1].cuda()
                    target_batches[step-1]= target_batches[step-1].cuda()
                    input_t[step-1] = input_t[step-1].cuda()
                    target_t[step-1] = target_t[step-1].cuda()
                # Run the train function
                loss_lda = train_mlp(
                        input_batches[step-1], input_lengths[step-1],
                        input_t[step-1], target_t[step-1],
                        encoder, decoder, mlp_topic, topic_choice,
                        lda_topic_opitimizer, batch_size
                )





