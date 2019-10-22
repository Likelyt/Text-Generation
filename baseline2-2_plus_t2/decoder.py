#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:09:21 2018

@author: liyt
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from Attn2 import Attn

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, num_topic, embedding_matrix, n_layers, dropout=0):#, max_length):
        super(AttnDecoderRNN, self).__init__()
        
        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.embedding_size = embedding_matrix.size(1)
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.num_topic = num_topic
        #self.max_length = max_length
        
        # Define layers
        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_pretrained = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.embedding_size, hidden_size, n_layers)
        self.topic_transofm = nn.Linear(num_topic, hidden_size)
        self.concat = nn.Linear(hidden_size * 3, hidden_size)
        self.out = nn.Linear(hidden_size, embedding_matrix.size(0))
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, target_topic, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        """
        input_seq: 1 x batch_size
        last_hidden: 1 x batch_size x hidden_size
        encoder_outputs: max_input_seq_len x batch_size x hidden_size
        """
        self.batch_size = input_seq.size(0)
        embedded = self.embedding_pretrained(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, self.batch_size, self.embedding_size) # 1 x B x N
        
        # Get current hidden state from input word and last hidden state
        # rnn_output: 1 x batch_size x hidden_size
        # hidden: 1 x batch_size x hidden_size
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights= self.attn(rnn_output, encoder_outputs)
        
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # batch_size x 1 x hidden_size

        target_topic_output = self.topic_transofm(target_topic)
        # Attentional vector using the RNN hidden state and context vector
        concat_input = torch.cat((rnn_output.squeeze(0), context.squeeze(1), target_topic_output), 1) # batch_size x 3*hidden_size
        concat_output = torch.tanh(self.concat(concat_input)) # batch_size x hidden_size
        

        # batch_size x vocab_output_size
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights



    


