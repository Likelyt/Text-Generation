
# Testing
import os
import torch
import sys
import numpy as np
import random_batch as rb
import random_batch_test as rbt
from evaluate_bleu import evaluate_randomly_bleu
from evaluate_ppx import evaluate_randomly_ppx
from score import score_calculator


def load_checkpoint(checkpoint_path):
    # It's weird that if `map_location` is not given, it will be extremely slow.
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


def val_load_checkpoint(optimal_epoch, val_data, 
                             data_vae_val,
                             h1_val_samples, h2_val_samples,
                             val_model_name, opt, LOAD_CHECKPOINT = True):

    if LOAD_CHECKPOINT:
        # load from path.
        checkpoint_path =  '%s%s%d%s' %(
            val_model_name,
            '_epoch_', optimal_epoch, 
            '.pt'
            )

        #checkpoint_path = os.path.join('model_result/', "Seq2seq_epoch_%d.pt" % optimal_epoch)
        checkpoint = load_checkpoint(checkpoint_path)
        encoder = checkpoint['encoder_model'].cuda()
        decoder = checkpoint['decoder_model'].cuda()
        model_vae_1 = checkpoint['vae_model'].cuda()
        TT = checkpoint['topic_transition_model'].cuda()
        input_voc = checkpoint['input_voc']
        output_voc = checkpoint['output_voc']
        encoder_hidden_size = checkpoint['encoder_hidden_size']
        decoder_hidden_size = checkpoint['decoder_hidden_size']
        learning_rate = checkpoint['learning_rate']
        start_epoch = checkpoint['epoch']
        batch_size = checkpoint['batch_size']



        # valing - Part
        print('\nval Epoch: %d' % (start_epoch))
        # Batches evaluate
        val_input_batches, val_input_lengths, val_target_batches, val_target_lengths = rbt.random_batch_testing(val_data, input_voc, output_voc, batch_size)
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
        for i in range(len(val_data)//batch_size):
            for l in range(batch_size):
                print('\nval Sample: %d' % (i*batch_size+l+1))
                print('\nInput Sentence: <', transform_from_index(val_input_batches[i][:,l], output_voc))
                print('Target Sentence: <', transform_from_index(val_target_batches[i][:,l], output_voc))
                print('Output Sentence: <', test_output_one_EOS[i][l])
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


        # Calculate Bleu Score    
        bleu_scores = [[] for t in range(len(val_input))]
        etp_scores = [[] for t in range(len(val_input))]
        div_scores = [[] for t in range(len(val_input))]
        for i in range(len(val_input)):
            temp_index = rb.batch_evaluate(val_output[i], output_voc, batch_size, USE_CUDA = True)
            bleu_scores[i], etp_scores[i], div_scores[i] = score_calculator(temp_index, val_target[i]) 

        etp_score = [sum(col) / float(len(col)) for col in zip(*etp_scores)]
        div_score = [sum(col) / float(len(col)) for col in zip(*div_scores)]
        blue_score = [sum(col) / float(len(col)) for col in zip(*bleu_scores)]
        print('\netp score: %.4f, %.4f, %.4f, %.4f'% (etp_score[0], etp_score[1], etp_score[2], etp_score[3]))
        print('Div score: %.4f, %.4f, %.4f, %.4f'% (div_score[0], div_score[1], div_score[2], div_score[3]))
        print('Bleu score: %.4f, %.4f, %.4f, %.4f' % (blue_score[0], blue_score[1], blue_score[2], blue_score[3]))
        print('Perplexity: %.4f\n' % (sum(ppx)/len(ppx)))
        sys.stdout.flush()       


    else:
        print('Need checkpoint, please try again')


