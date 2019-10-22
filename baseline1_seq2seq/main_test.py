#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 17:11:31 2018

@author: liyt
"""
import torch
import numpy as np
import random
import torch.nn as nn
from read_data_2 import read_data
from match_data_3 import Match_data_pair
import create_voc_4 as create_voc_4
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from train_epoch import trainIters
#from evaluate import evaluate_randomly
from testing import testing_load_checkpoint
from load_embedding_5 import load_embedding
import argparse



parser = argparse.ArgumentParser(description="main.py")
## Data options
parser.add_argument("-data_name", default='rest_data_pair_24K.txt', help='data name')
parser.add_argument("-review_name", default='rest', help='review type')

## Optimization options
parser.add_argument("-batch_size", type=int, default=128, help="Maximum batch size")
parser.add_argument("-hidden_size", type=int, default=256, help="hidden size")
parser.add_argument("-end_epoch", type=int, default=35, help="Epoch to stop training.")
parser.add_argument("-embedding_size", type=int, default=300, help="Size of entity embeddings")
parser.add_argument("-n_layers_encoder", type=int, default=2, help="layers of encoder")
parser.add_argument("-n_layers_decoder", type=int, default=1, help="layers of decoder")
parser.add_argument("-encoder_dropout", type=float, default=0.2, help="encoder dropout")
parser.add_argument("-bi", type=bool, default=True, help="direction of encoder")
parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")# 
parser.add_argument("-n_sample", type=int, default=10, help="Number of samples from Validation data output")

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
    text_all, l, sentence_0, sentence_1, sentence_2, sentence_3 = read_data('yelp_restaurant_review_all.json', 40000)
    #3. Create data to pairs
    data_0, data_pair = Match_data_pair(text_all, l, sentence_0, sentence_1, sentence_2, sentence_3)
    #4. Create Dictionary
    np.savetxt('yelp-data/data_pair_100K.txt', data_pair, delimiter=',', fmt = '%s') 
    '''
    data_pair = np.loadtxt('yelp-data/'+opt.data_name, delimiter=',', dtype = np.str).tolist()
    data, train_encoder, train_decoder, voc_input, voc_output = create_voc_4.build_voc(data_pair)
    embedding_matrix = load_embedding(data, voc_input)

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
    lr = opt.lr
    n_sample = opt.n_sample

    n = len(data) 
    n_train_ratio =  0.8
    n_val_ratio = 0.1
    n_test_ratio =  0.1
    
    # 6. Create Data batch
    train_data = data[0:int(n_train_ratio*n)] 
    val_data = data[int(n_train_ratio*n): int((n_train_ratio+n_val_ratio)*n)] 
    test_data = data[int((n_train_ratio+n_val_ratio)*n):n] 

    print('There are %d Test pairs' %(len(test_data)))
    test_tokens = 0
    for i in range(len(test_data)):
        test_tokens += len(test_data[i][0])
    print('There are %d Test tokens' %(test_tokens))

    '''
    # 7. Initialize Encoder, Decoder with attention mechanism
    encoder = EncoderRNN(voc_input.n_words, embedding_size, hidden_size, batch_size, bi, n_layers_encoder, dropout)
    #decoder = AttnDecoderRNN(attn_model, hidden_size, voc_output.n_words, n_layers_decoder, dropout)
    decoder = AttnDecoderRNN(attn_model, hidden_size, voc_output.n_words, embedding_matrix, n_layers_decoder)
    
    USE_CUDA = True
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()
    # 8.Train
    criterion = nn.NLLLoss()
    optimal_epoch, val_model_name = trainIters(train_data, val_data, 
        encoder, decoder, 
        voc_input, voc_output, 
        criterion, 
        train_epochs, batch_size, n_sample, opt, 
        print_every=10, plot_every=1, learning_rate=lr)

    print('\nThe optimal epoch is: %d ' %(optimal_epoch))
    '''

    optimal_epoch = 15
    val_model_name = 'model_result/rest_2241_hidden_size_256_E_layer_2_D_layer_1_Enc_bidi_1'
    testing_load_checkpoint(optimal_epoch, test_data, opt.review_name, val_model_name, True)
   
    






























