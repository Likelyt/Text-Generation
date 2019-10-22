#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 17:11:31 2018

@author: liyt
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
from read_data_2 import read_data
from match_data_3 import Match_data_pair
import create_voc_4_raw as create_voc_4
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from train_epoch import trainIters
from evaluate_bleu import evaluate_randomly_bleu
from topic_gene_lda import LDA_topic
import random_batch_topic as rbt
from MLP_topic_transition import Topic_Transition
from ntm_generation import VAE
from ntm_generation import NTM_topic_generation
from validation import val_load_checkpoint
from load_embedding_5 import load_embedding


parser = argparse.ArgumentParser(description="main.py")
## Data options
parser.add_argument("-data_name", default='rest_data_pair_24K.txt', help='data name')
parser.add_argument("-review_name", default='rest', help='review type')

## Optimization options
parser.add_argument("-batch_size", type=int, default=128, help="Maximum batch size")
parser.add_argument("-hidden_size", type=int, default=256, help="hidden size")
parser.add_argument("-end_epoch", type=int, default=50, help="Epoch to stop training.")
parser.add_argument("-embedding_size", type=int, default=300, help="Size of entity embeddings")
parser.add_argument("-n_layers_encoder", type=int, default=2, help="layers of encoder")
parser.add_argument("-n_layers_decoder", type=int, default=1, help="layers of decoder")
parser.add_argument("-encoder_dropout", type=float, default=0.2, help="encoder dropout")
parser.add_argument("-bi", type=bool, default=True, help="direction of encoder")
parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("-n_sample", type=int, default=10, help="Number of samples from Validation data output")
parser.add_argument("-n_topic", type=int, default=20, help="Number of topic number: J")
parser.add_argument("-h1_train_samples", type=int, default=1, help="Number of h1 samples in training")
parser.add_argument("-h2_train_samples", type=int, default=1, help="Number of h2 samples in training")
parser.add_argument("-h1_testing_samples", type=int, default=1, help="Number of h1 samples in training")
parser.add_argument("-h2_testing_samples", type=int, default=10, help="Number of h2 samples in training")


parser.add_argument("-torch_seed", type=int, default=12345, help="Random seed for pytorch")
parser.add_argument("-optimal_epoch", type=int, default=16, help="optimal_epoch for testing and validation")


opt = parser.parse_args()
print(opt)


if __name__ == "__main__":
    torch.cuda.manual_seed(opt.torch_seed)
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #2. Read Data
    text_all, l, sentence_0, sentence_1, sentence_2, sentence_3 = read_data('yelp_restaurant_review_all.json', 10000)
    #3. Create data to pairs
    data_0, data_pair = Match_data_pair(text_all, l, sentence_0, sentence_1, sentence_2, sentence_3)
    #4. Create Dictionary
    np.savetxt('yelp-data/data_pair_100K.txt', data_pair, delimiter=',', fmt = '%s') 
    '''
    data_pair = np.genfromtxt('yelp-data/'+opt.data_name, delimiter=',', dtype = np.unicode_, encoding='utf8').tolist()
    data, keep_index, train_encoder, train_decoder, voc_input, voc_output = create_voc_4.build_voc(data_pair)
    # Save keep_index of 24K raw data
    # np.savetxt('yelp-data/keep_index_24K.txt', keep_index, delimiter=',', fmt = '%d')
    print('There are %d pairs sentence in total' % (len(data)))
    embedding_matrix = load_embedding(data, voc_input)

    # 5. Hyper-parameters setting
    hidden_size = opt.hidden_size
    vae_model_hidden_size = 100
    embedding_size = opt.embedding_size
    n_layers_encoder = opt.n_layers_encoder
    n_layers_decoder = opt.n_layers_decoder    
    bi = opt.bi
    if n_layers_encoder == 2:
        dropout = opt.encoder_dropout
    else:
        dropout = 0


    attn_model = 'concat' # Fixed
    batch_size= opt.batch_size
    train_epochs = opt.end_epoch
    lr = opt.lr
    n_sample = opt.n_sample
    n = len(data) 

    number_of_topic = opt.n_topic
    h1_samples = opt.h1_train_samples
    h2_samples = opt.h2_train_samples
    h1_val_samples = opt.h1_testing_samples
    h2_val_samples = opt.h2_testing_samples

    n_train_ratio =  0.8
    n_val_ratio = 0.1
    n_test_ratio =  0.1


    # 6. Create Data batch
    train_data = data[0:int(len(data)/batch_size*n_train_ratio)*batch_size] 
    val_data = data[int(len(data)/batch_size*n_train_ratio)*batch_size: int(len(data)/batch_size*(n_train_ratio+n_val_ratio))*batch_size] 
    test_data = data[int(len(data)/batch_size*(n_train_ratio+n_val_ratio))*batch_size:n] 
    


    data_vae, input_size = NTM_topic_generation(opt.data_name, number_of_topic,  batch_size)
    data_vae_train = data_vae[0:int(len(data)/batch_size*n_train_ratio)]
    data_vae_val = data_vae[int(len(data)/batch_size*n_train_ratio): int(len(data)/batch_size*(n_train_ratio+n_val_ratio))]
    data_vae_test = data_vae[int(len(data)/batch_size*(n_train_ratio+n_val_ratio)):int(n/batch_size)] 

    '''
    # 7. Initialize Encoder, Decoder with attention mechanism
    encoder = EncoderRNN(voc_input.n_words, embedding_size, hidden_size, batch_size, bi, n_layers_encoder, dropout=dropout)
    decoder = AttnDecoderRNN(attn_model, hidden_size, voc_output.n_words, number_of_topic, embedding_matrix ,n_layers_decoder, dropout)

    model_vae_1 = VAE(input_size, vae_model_hidden_size, number_of_topic)
    TT = Topic_Transition(hidden_size+number_of_topic, hidden_size, number_of_topic)
    print(encoder)
    print(decoder)
    print(model_vae_1)
    print(TT)


    USE_CUDA = True
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()
        model_vae_1.cuda()
        TT.cuda()

    # 8.Train
    criterion = nn.NLLLoss()

    optimal_epoch, val_model_name =  trainIters(train_data, val_data, test_data, 
                                        data_vae_train, data_vae_val, data_vae_test, 
                                        encoder, decoder, model_vae_1, TT,
                                        h1_samples, h2_samples, 
                                        h1_val_samples, h2_val_samples,
                                        voc_input, voc_output, 
                                        criterion, train_epochs, batch_size, n_sample, number_of_topic, opt,
                                        print_every=10, plot_every=1, learning_rate=lr, USE_CUDA=True)
    '''
    optimal_epoch = opt.optimal_epoch
    print('\nThe optimal epoch is: %d ' %(optimal_epoch))
    val_model_name = 'model_result/rest_2314_hidden_size_256_E_layer_2_D_layer_1_E_bi_1_T_20_L1_1_L2_10'
    val_load_checkpoint(
            optimal_epoch, test_data, 
            data_vae_test, 
            opt.h1_testing_samples, opt.h2_testing_samples,
            val_model_name, opt)







































