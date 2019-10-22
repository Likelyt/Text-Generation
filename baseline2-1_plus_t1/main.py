#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 17:11:31 2018

@author: liyt
"""

import torch
import numpy as np
import argparse
import torch.nn as nn
from read_data_2 import read_data
from match_data_3 import Match_data_pair
import create_voc_4_raw as create_voc_4
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from train_epoch import trainIters
from evaluate import evaluate_randomly
from topic_gene_lda import LDA_topic
import random_batch_topic as rbt
from MLP_topic_transition import Topic_Transition
from ntm_generation import NTM_topic_generation
from load_embedding_5 import load_embedding
from testing imoirt testing_load_checkpoint


parser = argparse.ArgumentParser(description="main.py")
## Data options
parser.add_argument("-data_name", default='rest_data_pair_24K.txt', help='data name')
parser.add_argument("-review_name", default='rest', help='review type')
parser.add_argument("-topic_choice", default='NTM', help='Neural topic model')


## Optimization options
parser.add_argument("-batch_size", type=int, default=128, help="Maximum batch size")
parser.add_argument("-hidden_size", type=int, default=256, help="hidden size")
parser.add_argument("-end_epoch", type=int, default=40, help="Epoch to stop training.")
parser.add_argument("-embedding_size", type=int, default=300, help="Size of entity embeddings")
parser.add_argument("-n_layers_encoder", type=int, default=2, help="layers of encoder")
parser.add_argument("-n_layers_decoder", type=int, default=1, help="layers of decoder")
parser.add_argument("-encoder_dropout", type=float, default=0.2, help="encoder dropout")
parser.add_argument("-bi", type=bool, default=True, help="direction of encoder")
parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("-n_sample", type=int, default=10, help="Number of samples from Validation data output")


##Topic model options
parser.add_argument("-topic_dimension", type=int, default=10, help="Number of topics J")
parser.add_argument("-topic_hidden_size", type=int, default=400, help="hidden size of topic model")
parser.add_argument("-topic_train_epoch", type=int, default=50, help="Number of training epochs of topic model")


## Random seed
parser.add_argument("-torch_seed", type=int, default=12345, help="Random seed for pytorch")
parser.add_argument("-numpy_seed", type=int, default=12345, help="Random seed for numpy")


opt = parser.parse_args()
print(opt)


if __name__ == "__main__":
    torch.cuda.manual_seed(opt.torch_seed)
    np.random.seed(opt.numpy_seed)
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #2. Read Data
    text_all, l, sentence_0, sentence_1, sentence_2, sentence_3 = read_data('yelp_restaurant_review_all.json', 10000)
    #3. Create data to pairs
    data_0, data_pair = Match_data_pair(text_all, l, sentence_0, sentence_1, sentence_2, sentence_3)
    #4. Create Dictionary
    np.savetxt('yelp-data/data_pair_100K.txt', data_pair, delimiter=',', fmt = '%s') 
    '''
    data_pair = np.loadtxt('yelp-data/'+opt.data_name, delimiter=',', dtype = np.str).tolist()
    data, keep_index, train_encoder, train_decoder, voc_input, voc_output = create_voc_4.build_voc(data_pair)
    embedding_matrix = load_embedding(data, voc_input)
    # Save keep_index of 24K raw data
    # np.savetxt('yelp-data/keep_index_24K.txt', keep_index, delimiter=',', fmt = '%d')
    print('There are %d pairs sentence in total' % (len(data)))

    # 5. Hyper-parameters setting
    hidden_size = opt.hidden_size
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
    #mlp_epoch = opt.mlp_epoch
    lr = opt.lr
    n_sample = opt.n_sample
    n = len(data) 
    n_train_ratio =  0.8
    n_val_ratio = 0.1
    n_test_ratio =  0.1
    topic_choice = opt.topic_choice
    

    if topic_choice == 'NTM':
        '''
        ntm_topic_dim = opt.topic_dimension  #10
        number_of_topic = ntm_topic_dim
        ntm_h_dim = opt.topic_hidden_size  #400
        ntm_num_epochs = opt.topic_train_epoch # 5
        ntm_batch_size = opt.batch_size
        ntm_lr = opt.lr/2
        sentence_topic_distribution = NTM_topic_generation(opt.data_name, ntm_topic_dim, ntm_h_dim, ntm_num_epochs, ntm_batch_size, opt,ntm_lr).tolist()
        '''
        # Or load
        load_topic_name = '%s%s%s%s%s%d%s%s%s' % ('latent_topic_distribution/',opt.review_name,'_',opt.topic_choice,'_z_',opt.topic_dimension,'_',len(data),'.txt')
        sentence_topic_distribution = np.loadtxt(load_topic_name, delimiter=' ', dtype = 'float').tolist()
        '''
        ntm_input_size = hidden_size + ntm_topic_dim
        ntm_hidden_size = 300
        ntm_output_size = ntm_topic_dim
        ntm_topic = Topic_Transition(ntm_input_size, ntm_hidden_size, ntm_output_size)
        
    else:
        # MLP - LDA - Topic - Transition
        number_of_topic_lda = 5
        number_of_topic = number_of_topic_lda
        lda_input_size = hidden_size + number_of_topic
        lda_hidden_size = 300
        lda_output_size = number_of_topic
        sentence_topic_distribution, beta_matrix = LDA_topic(number_of_topic_lda)
        lda_topic = Topic_Transition(lda_input_size, lda_hidden_size, lda_output_size)
    '''

    #Wrap data with topic distribution
    data = rbt.wrapper_data_with_topic(data, sentence_topic_distribution, topic_choice)

    # 6. Create Data batch
    train_data = data[0:int(n_train_ratio*n)] 
    val_data = data[int(n_train_ratio*n): int((n_train_ratio+n_val_ratio)*n)] 
    test_data = data[int((n_train_ratio+n_val_ratio)*n):n] 
    train_topic = sentence_topic_distribution[0:int(n_train_ratio*n)] 
    val_topic = sentence_topic_distribution[int(n_train_ratio*n): int((n_train_ratio+n_val_ratio)*n)] 
    test_topic = sentence_topic_distribution[int((n_train_ratio+n_val_ratio)*n):n] 

    '''
    # 7. Initialize Encoder, Decoder with attention mechanism
    encoder = EncoderRNN(voc_input.n_words, embedding_size, hidden_size, batch_size, bi, n_layers_encoder, dropout=dropout)
    decoder = AttnDecoderRNN(attn_model, hidden_size, voc_output.n_words, opt.topic_dimension, embedding_matrix, n_layers_decoder, dropout)
    
    USE_CUDA = True
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()


    # 8.Train
    criterion = nn.NLLLoss()

    if topic_choice == 'LDA':
        trainIters(train_data, val_data, encoder, decoder, lda_topic, topic_choice,
                voc_input, voc_output, 
                criterion, opt.end_epoch, batch_size, n_sample, mlp_epoch,
                print_every=10, plot_every=1, learning_rate=lr, USE_CUDA=True)
    else:
        optimal_epoch, val_model_name = trainIters(train_data, val_data, encoder, decoder, topic_choice,
                voc_input, voc_output, 
                criterion, opt.end_epoch, batch_size, n_sample, opt,
                print_every=10, plot_every=1, learning_rate=lr, USE_CUDA=True)
    

    print('\nThe optimal epoch is: %d ' %(optimal_epoch))
    '''


    optimal_epoch = 14
    val_model_name = 'model_result/rest_2241_hidden_size_256_E_layer_2_D_layer_1_Enc_bidi_1'
    testing_load_checkpoint(optimal_epoch, test_data, opt.review_name, val_model_name, opt, True)
    


















