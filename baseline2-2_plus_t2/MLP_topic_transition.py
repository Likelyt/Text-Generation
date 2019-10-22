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
        self.middle = nn.Linear(hidden_size, hidden_size)
        self.f2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x, encoder_vec):
        X = torch.cat((encoder_vec.squeeze(0), x), 1)
        z = self.f1(X)
        z = self.relu(z)
        z = self.middle(z)
        z = self.relu(z)
        z = self.f2(z)
        y = F.softmax(z, dim = 1)
        return y


#batch_size= 128
#input_size = 5
#hidden_size = 100
#output_size = 5
#learning_rate = 0.001
#num_epochs = 20#

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#topic = sentence_topic_distribution
#data_size_x = len(data)
#train_batches_x = utils.create_batch(data_size_x, batch_size, shuffle=False)
#data_loader_x, data_loader_y  = utils.fetch_topic_data(data, train_batches_x, batch_size)
#model = Topic_Transition_lda(input_size, hidden_size, output_size).to(device)
#criterion = nn.CrossEntropyLoss()  
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)#

#for epoch in range(num_epochs):
#	for i in range(len(data_loader_x)):
#		x = torch.tensor(data = data_loader_x[i], device=device).float()
#		y = torch.tensor(data = data_loader_y[i], device=device).float()
#		y_hat = model(x)
#		log_loss = -torch.sum(torch.mul(torch.log(y_hat), y), dim = 1)
#		loss = torch.sum(log_loss)/batch_size
#		optimizer.zero_grad()
#		loss.sum().backward()
#		optimizer.step()
#		if (i+1)%100 == 0:
#			print("Epoch[{}/{}], Step [{}/{}], Log Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(data_loader_x[i]), loss))#

#for i in range(len(data_loader_x)-1):
#	x = torch.tensor(data = data_loader_x[i], device=device).float()
#	y = torch.tensor(data = data_loader_y[i], device=device).float()
#	y_hat = model(x)
#	log_loss = -torch.sum(torch.mul(torch.log(y_hat), y), dim = 1)
#	loss = torch.sum(log_loss)/batch_size
#	optimizer.zero_grad()
#	loss.sum().backward()
#	optimizer.step()
#	if (i+1)%1 == 0:
#		print("Epoch[{}/{}], Step [{}/{}], Log Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(data_loader_x), loss))














