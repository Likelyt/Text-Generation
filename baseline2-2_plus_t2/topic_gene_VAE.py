import os
import numpy as np
import utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
#from torchvision import transforms

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer


class VAE(nn.Module):
    def __init__(self, input_size, h_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim,input_size)
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)
    def parameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std  
    def decode(self, h):
        z = F.softmax(h, dim = 1)
        y = F.softmax(self.fc3(z), dim = 1)
        return z, y 
    def forward(self, x):
        mu, log_var = self.encode(x)
        h = self.parameterize(mu, log_var)
        z, y = self.decode(h)
        return y, mu, log_var, z