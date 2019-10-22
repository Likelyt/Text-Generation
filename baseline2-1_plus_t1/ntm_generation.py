import os
import sys
import numpy as np
import utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import create_voc_4_raw as create_voc_4

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer


def MaxLength(text):
	# Suppose input data is list of sentence
    max_length = 0
    for i in range(len(text)):
        if max_length < len(text[i].split()):
            max_length = len(text[i].split())
    return max_length

def AvgLength(text):
	# Suppose the input data is pair format:
	avg_length_encode = 0
	avg_length_decode = 0
	for i in range(len(text)):
		avg_length_encode += len(text[i][0].split())
		avg_length_decode += len(text[i][1].split())
	return avg_length_encode/len(text), avg_length_decode/len(text)


def Remove_duplicate(text):
	n = len(text)
	voc_size = len(text[0])
	# Keep first sentence and last sentence
	text_clean = torch.Tensor(int(n/2+1),voc_size)
	text_clean[0] = text[0]
	for i in range(1, n, 2):
		text_clean[int((i-1)/2+1)] = text[i]
	return text_clean



class VAE(nn.Module):
    def __init__(self, input_size, h_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, input_size)
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





def NTM_topic_generation(data_name, z_dim, h_dim, num_epochs, batch_size, opt, lr = 0.0001):
    # 0. Load Data Pairs
    data_pair = np.loadtxt('yelp-data/'+ data_name, delimiter=',', dtype = np.str).tolist()
    data, keep_index, train_encoder, train_decoder, voc_input, voc_output = create_voc_4.build_voc(data_pair)

    # 1. Transform data to one-hot 
    T = Tokenizer()
    # Odd number is input sentence, even number is target sentence
    data_one_column = [] 
    for i in range(len(data)):
        data_one_column.append(data[i][0])
    data_one_column.append(data[i][1]) # all sentence 22411

    T.fit_on_texts(data_one_column)
    one_hot_docs = T.texts_to_matrix(data_one_column, mode = 'count') 
    #print('There are %d words in total' % (len(one_hot_docs[0])))
    vocab_size = len(T.word_index) + 1 # 5441

    encoded_docs = T.texts_to_sequences(data_one_column)
    max_length = MaxLength(data_one_column) # 111
    encoder_avg_len, decoder_avg_len = AvgLength(data_pair)
    print('The max length of sentence is: %d' % (max_length))
    print('The encoder sentences average length is: %.2f\nThe decoder sentences average length is: %.2f' % (encoder_avg_len, decoder_avg_len))

    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


    embeddings_index = dict()
    f = open('Glove/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()

    # create a weight matrix for words in training docs
    Glove_dim = 300
    embedding_matrix = np.zeros((vocab_size, Glove_dim))  #(5441, 300)
    for word, i in T.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    e = Embedding(vocab_size, Glove_dim, weights=[embedding_matrix], input_length=max_length, trainable=False) # 300 is output dimension

    input_size = vocab_size # 5411 
    learning_rate = lr
    shf = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    torch_onehot_docs = torch.from_numpy(one_hot_docs) # 22411, 5441
    # Remove duplicates
    #torch_onehot_docs = Remove_duplicate(torch_onehot_doc)
    data_size = len(torch_onehot_docs)
    train_batches = utils.create_batch(data_size, batch_size, shuffle=shf)
    data_loader = utils.fetch_data(torch_onehot_docs, train_batches)
    latent_z = []


    #train NTM model
    model = VAE(input_size, h_dim, z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.99, 0.999))

    for epoch in range(num_epochs):
        for i, x in data_loader.items():
            x = torch.tensor(data = x, device=device).float()
            y_hat, mu, log_var, z = model(x)
            # Compute reconstruction loss and kl divergence 
            log_doc_loss = -torch.sum(torch.mul(torch.log(y_hat), x), dim = 1)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) -  log_var.exp())
            # Backprop and optimize
            log_doc_loss_ave = torch.sum(log_doc_loss)/batch_size
            loss = torch.sum(log_doc_loss)/batch_size + kl_div
            if epoch == num_epochs-1:
                latent_z.append(z)
            optimizer.zero_grad()
            loss.sum().backward()
            #loss.backward()
            optimizer.step()
            if (i+1)%10 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Log Doc Loss: {:.4f}, KL Div: {:.5f}"
                      .format(epoch+1, num_epochs, i+1, len(data_loader), log_doc_loss_ave, kl_div.item()))


    # Write latent topic distribution into txt files
    latent_z_np = list()
    for i in range(len(latent_z)):
        latent_z_np.append(latent_z[i].cpu().detach().numpy())
    latent_z_np_stack = np.vstack(latent_z_np)[0:data_size]

    save_name = '%s%s%s%s%s%d%s%d%s' % (
        'latent_topic_distribution/', opt.review_name,
        '_',opt.topic_choice,
        '_z_',opt.topic_dimension, 
        '_',len(data),
        '.txt'
        )

    np.savetxt(save_name,latent_z_np_stack, fmt='%.4e', delimiter=' ')

    return latent_z_np_stack








