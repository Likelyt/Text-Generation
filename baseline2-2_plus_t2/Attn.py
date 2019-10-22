#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 18:32:43 2018

@author: liyt
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.ones(1, hidden_size))

    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.mm(encoder_output.transpose(0,1))[0]
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.mm(energy.transpose(0,1))[0]
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.mm(energy.transpose(0,1))[0]
            return energy
        

    def forward(self, hidden, encoder_outputs, USE_CUDA = True):
        """
        Encoder_outputs: max_encoder_seq_len x batch_size x hidden_size
        hidden: 1 x batch_size x hidden_size
        """
        max_len = encoder_outputs.size(0) # encoder_max_len
        this_batch_size = encoder_outputs.size(1) # batch_size

        # Create variable to store attention energies
        attn_energies = torch.zeros(this_batch_size, max_len).float() # Batch_size x encoder_max_len

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        start = time.time()

        for b in range(this_batch_size):
            start_b = time.time() 
            # Calculate energy for each encoder output
            # 
            # 
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))


            end_b = time.time()
            print('Attn one batch time: %.4f' % (end_b-start_b))
        end = time.time()
        print('Attn all batch time: %.4f' % (end-start))
        

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim = 1).unsqueeze(1)
    

        






