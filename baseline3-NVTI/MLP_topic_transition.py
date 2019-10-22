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
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, output_size)
        self.fc22 = nn.Linear(hidden_size, output_size)
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc21(h)
        log_var = self.fc22(h)
        # Clapm the variance 
        clamp_log_var = torch.clamp(log_var, max = 4.0)
        return mu, clamp_log_var
    def parameterize(self, mu, log_var, h2_samples):
        std = torch.exp(log_var/2)
        if h2_samples == 1:
            epsilon = torch.randn_like(std)
            h = mu + epsilon * std 
            h = F.softmax(h, dim = 1)
        else:
            epsilon = torch.randn((h2_samples*std.size(0), std.size(1))).cuda().float()
            epsilon_list = torch.split(epsilon, std.size(0), dim=0) # it returns a tuple
            h = []
            for i in range(h2_samples):
                h.append(mu + epsilon_list[i] * std)
            for i in range(h2_samples):
                h[i] = F.softmax(h[i], dim = 1)
        return h
    def forward(self, t_1, encoder_vec, h2_samples):
        t_1 = torch.cat((encoder_vec.squeeze(0), t_1), 1)
        mu, log_var = self.encode(t_1)
        h = self.parameterize(mu, log_var, h2_samples)
        z = h
        return mu, log_var, z