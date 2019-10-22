import os
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Topic_Transition(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Topic_Transition, self).__init__()
        self.f1 = nn.Linear(input_size, hidden_size)
        self.middle = nn.Linear(hidden_size, 10*hidden_size)
        self.f2 = nn.Linear(hidden_size*10, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x, encoder_vec):
        X = torch.cat((encoder_vec.squeeze(0), x), 1)
        z = self.f1(X)
        z = self.tanh(z)
        z = self.middle(z)
        z = self.tanh(z)
        z = self.f2(z)
        y = F.softmax(z, dim = 1)
        return y




def evaluate_TT(val_input_batches, val_input_lengths, val_input_t, val_target_t,  
                encoder, decoder, TT, 
                batch_size):

    encoder.train(False)
    decoder.train(False)
    TT.train(False)

    encoder_outputs, encoder_hidden = encoder(val_input_batches, val_input_lengths, None)
    y_hat = TT(val_input_t, encoder_hidden[:decoder.n_layers])
    log_loss = -torch.sum(torch.mul(torch.log(y_hat), val_target_t), dim = 1)
    loss_TT = torch.sum(log_loss)/batch_size
    #print("Val Topic transition: Log Loss: {:.4f}".format(loss_TT))


    encoder.train(True)
    decoder.train(True)
    TT.train(True)

    return loss_TT
    
















