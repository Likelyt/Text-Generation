#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:13:01 2018

@author: liyt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, batch_size, bi, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.bi = bi

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional = bi)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        # input_lengths and input_seq have already been sorted
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding: Pads a packed batch of variable length sequences.
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        if self.bi == True:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
    def init_hidden(self):
        # variable of size [num_layers*num_directions, b_sz, hidden_sz]
        return torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
