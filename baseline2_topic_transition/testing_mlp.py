# Testing
import os
import torch
import sys
import numpy as np
import random_batch as rb
import random_batch_topic as rbt
from evaluate_bleu import evaluate_randomly_bleu
from evaluate_ppx import evaluate_randomly_ppx
from evaluate_bleu import transform_from_index
from score import score_calculator
from train import train_mlp
from MLP_topic_transition import evaluate_TT


def load_checkpoint(checkpoint_path):
    # It's weird that if `map_location` is not given, it will be extremely slow.
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

def classify(a):
    batch_size, dim = a.size()
    for i in range(batch_size):
            a[i] = torch.where(a[i] == max(a[i]), torch.tensor(1), torch.tensor(0))
    return a



def testing_load_checkpoint_mlp(optimal_epoch, train_data, val_data, test_data,
    review_type, val_model_name, opt, TT,
    LOAD_CHECKPOINT = True, USE_CUDA = True):

    if LOAD_CHECKPOINT:
        # load from path.
        checkpoint_path =  '%s%s%d%s%d%s' %(
            val_model_name,'_T_', opt.topic_dimension, '_epoch_', optimal_epoch, '.pt'
            )

        checkpoint = load_checkpoint(checkpoint_path)
        encoder = checkpoint['encoder_model'].cuda()
        decoder = checkpoint['decoder_model'].cuda()
        input_voc = checkpoint['input_voc']
        output_voc = checkpoint['output_voc']
        encoder_hidden_size = checkpoint['encoder_hidden_size']
        decoder_hidden_size = checkpoint['decoder_hidden_size']
        learning_rate = checkpoint['learning_rate']
        start_epoch = checkpoint['epoch']
        batch_size = checkpoint['batch_size']


        TT_optimizer = torch.optim.Adam(TT.parameters(), lr = learning_rate, betas=(0.99, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        train_mlp_epoch = 0
        loss_mlp = 0
        mlp_loss_total = []
        while train_mlp_epoch < opt.mlp_epoch:
            train_mlp_epoch += 1

            # Get training data for this cycle
            input_batches, input_lengths, target_batches, target_lengths, input_t, target_t = rbt.random_batch(train_data, input_voc, output_voc, batch_size)
            for step in range(1, len(train_data)//batch_size+1):
                # Trans data to GPU mode
                if USE_CUDA:
                    input_batches[step-1] = input_batches[step-1].cuda()
                    target_batches[step-1]= target_batches[step-1].cuda()
                    input_t[step-1] = input_t[step-1].cuda()
                    #target_t[step-1] = target_t[step-1].cuda()

                #target_t[step-1] = classify(target_t[step-1])

                loss_mlp = train_mlp(input_batches[step-1], input_lengths[step-1], input_t[step-1], target_t[step-1].cuda(), 
                    encoder, decoder, TT,
                    opt.topic_choice,
                    TT_optimizer, batch_size)


            # Validation - Part
            print('\nVal Epoch: %d\n' % (train_mlp_epoch))

            val_input_batches, val_input_lengths, val_target_batches, val_target_lengths, val_input_t, val_target_t = rbt.random_batch(val_data, input_voc, output_voc, batch_size)

            val_loss_TT = [[] for _ in range(len(val_data)//batch_size)]
            for i in range(len(val_data)//batch_size):
                if USE_CUDA:
                    val_input_batches[i] = val_input_batches[i].cuda()
                    val_input_t[i] = val_input_t[i].cuda()
                    val_target_t[i] = val_target_t[i].cuda()

                val_loss_TT[i] = evaluate_TT(val_input_batches[i], val_input_lengths[i], val_input_t[i], val_target_t[i],  
                                encoder, decoder, TT, 
                                batch_size)

            print('\nVal MLP Loss: %.4f\n' % (sum(val_loss_TT)/len(val_loss_TT)))

            mlp_loss_total.append(sum(val_loss_TT)/len(val_loss_TT))

            sys.stdout.flush()           

            # Check points
            checkpoint_TT = {
                'TT': TT,
            }

            val_TT_name = '%s%s_%d%s%d' % (
                'model_result/TT_', opt.review_name, len(val_data),
                '_TZ_',opt.topic_dimension
                )
            val_TT_epoch = '%s%s%d%s' %(
                val_TT_name, '_epoch_', train_mlp_epoch, '.pt'
                )

            # Need change the name of saving the model parameter name
            #torch.save(checkpoint_TT, val_TT_epoch)
            print("Save model as %s" % val_TT_epoch)





        MLP_optimal_epoch = mlp_loss_total.index(min(mlp_loss_total))+1

        print(MLP_optimal_epoch)

        return MLP_optimal_epoch
    else:
        print('Need checkpoint, please try again')








        '''
        # Testing - Part
        print('\nTest Epoch: %d' % (start_epoch))
        # Batches evaluate
        test_input_batches, test_input_lengths, test_target_batches, test_target_lengths, test_input_t, test_target_t = rbt.random_batch(test_data, input_voc, output_voc, batch_size)
        test_input = [[] for _ in range(len(test_data)//batch_size)]
        test_output = [[] for _ in range(len(test_data)//batch_size)]
        test_output_one_EOS = [[] for _ in range(len(test_data)//batch_size)]
        test_target = [[] for _ in range(len(test_data)//batch_size)]
        ppx = [[] for _ in range(len(test_data)//batch_size)]


        for i in range(len(test_data)//batch_size):
            test_input[i], test_output[i], test_output_one_EOS[i], test_target[i], ppx[i] = evaluate_randomly_ppx(test_input_batches[i], test_input_lengths[i], test_target_batches[i], test_target_lengths[i],
                test_input_t[i], test_target_t[i], 
                batch_size, encoder, decoder, input_voc, output_voc)

        # Print some results
        for i in range(len(test_data)//batch_size):
            for l in range(batch_size):
                print('\ntest Sample: %d' % (i*batch_size+l+1))
                print('\nInput Sentence: <', transform_from_index(test_input_batches[i][:,l], output_voc))
                print('Target Sentence: <', transform_from_index(test_target_batches[i][:,l], output_voc))
                print('Output Sentence: <', test_output_one_EOS[i][l])
        sys.stdout.flush()   



        # Compute Blue Score
        test_input = [[] for _ in range(len(test_data)//batch_size)]
        test_output = [[] for _ in range(len(test_data)//batch_size)]
        test_output_one_EOS = [[] for _ in range(len(test_data)//batch_size)]
        test_target = [[] for _ in range(len(test_data)//batch_size)]

        for i in range(len(test_data)//batch_size):
            test_input[i], test_output[i], test_output_one_EOS[i], test_target[i] = evaluate_randomly_bleu(test_input_batches[i], test_input_lengths[i], test_target_batches[i], test_target_lengths[i],
                test_input_t[i], test_target_t[i], 
                batch_size, encoder, decoder, input_voc, output_voc)




        # Calculate Bleu Score    
        bleu_scores = [[] for t in range(len(test_input))]
        etp_scores = [[] for t in range(len(test_input))]
        div_scores = [[] for t in range(len(test_input))]
        for i in range(len(test_input)):
            temp_index = rb.batch_evaluate(test_output[i], output_voc, batch_size, USE_CUDA = True)
            bleu_scores[i], etp_scores[i], div_scores[i] = score_calculator(temp_index, test_target[i]) 

        etp_score = [sum(col) / float(len(col)) for col in zip(*etp_scores)]
        div_score = [sum(col) / float(len(col)) for col in zip(*div_scores)]
        blue_score = [sum(col) / float(len(col)) for col in zip(*bleu_scores)]
        print('etp score: %.4f, %.4f, %.4f, %.4f'% (etp_score[0], etp_score[1], etp_score[2], etp_score[3]))
        print('Div score: %.4f, %.4f, %.4f, %.4f'% (div_score[0], div_score[1], div_score[2], div_score[3]))
        print('Bleu score: %.4f, %.4f, %.4f, %.4f' % (blue_score[0], blue_score[1], blue_score[2], blue_score[3]))
        print('Perplexity of Test Set: %.4f\n' % (sum(ppx)/len(ppx)))
        sys.stdout.flush()       
        '''

        '''
        testing_checkpoint = {
            'etp_score': etp_score,
            'div_score': div_score,
            'blue_score': blue_score, 
            'optimal_epoch': optimal_epoch,
            'hidden_size':encoder_hidden_size,
            'encoder_n_layers': encoder.n_layers,
            'decoder_n_layers': decoder.n_layers,
            'encoder_bidireaction':encoder.bi,
        }
        testing_model_name = '%s%s_%d%s%d%s%d%s%d%s%d%s%s%s%d%s' % (
            'testing_result/score_', review_type ,len(test_data),
            '_hidden_size_', encoder_hidden_size,
            '_optimal_epoch_',optimal_epoch,
            '_E_layer_', encoder.n_layers,
            '_D_layer_', decoder.n_layers,
            '_E_bi_',encoder.bi,
            '_T_',opt.topic_dimension,
            '.pt'
            )
        torch.save(testing_checkpoint, testing_model_name)

    else:
        print('Need checkpoint, please try again')

    '''




 