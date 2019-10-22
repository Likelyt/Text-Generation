import os
import numpy as np
import utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F


from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer


class VAE(nn.Module):
    def __init__(self, input_size, h_dim, z_dim, h1_samples):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim,input_size)
        self.h1_samples = h1_samples
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc21(h)
        log_var = self.fc22(h)
        return mu, log_var
    def parameterize(self, mu, log_var, h1_samples):
        std = torch.exp(log_var/2)
        if h1_samples == 1:
            epsilon = torch.randn_like(std)
            h = mu + epsilon * std 
        else:
            epsilon = torch.randn((h1_samples*std.size(0), std.size(1)), device=device).float()
            epsilon_list = torch.split(epsilon, std.size(0), dim=0) # it returns a tuple
            h = []
            for i in range(h1_samples):
                h.append(mu + epsilon_list[i] * std)
        return h
    def decode(self, h, h1_samples):
        if h1_samples == 1:
            z = F.softmax(h, dim = 1)
            y = F.softmax(self.fc3(z), dim = 1)
        else:
            z = []
            y = []
            for i in range(h1_samples):
                z.append(F.softmax(h[i], dim = 1))
                y.append(F.softmax(self.fc3(z[i]), dim = 1))
        return z, y 
    def forward(self, x):
        mu, log_var = self.encode(x)
        h = self.parameterize(mu, log_var, self.h1_samples)
        z, y = self.decode(h, self.h1_samples)
        return y, mu, log_var, z


