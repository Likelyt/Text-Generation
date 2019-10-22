import json
import os
import torch
import random
import numpy as np

def write_data_json(review, n_select, name):
	data = review[0:n_select]
	jsonData = json.dumps(data)
	data_name = name+'.json'
	data_name = os.path.join(os.getcwd()+'/yelp-data/', data_name)   
	with open(data_name, 'w') as f:
		json.dump(jsonData, f)
	return data


def read_data_txt(name):
	with open(name, 'r') as f:
		data = f.readlines()
	return data

def clean_doc(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def create_batch(data_size, batch_size, shuffle=True):
    batches = []
    ids = list(range(data_size))
    if shuffle:
        random.shuffle(ids)
    for i in range(data_size // batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])
    # the batch of which the length is less than batch_size
    rest = data_size % batch_size
    if rest > 0:
        batches.append(ids[-rest:] + [-1] * (batch_size - rest))
    return batches


def fetch_data(data, idx_batch):
    x = {}
    for i in range(len(idx_batch)):
        x[i] = data[idx_batch[i]]
    return x
      

def fetch_topic_data(data, idx_batch, batch_size):
    x = [[] for _ in range(len(idx_batch))]
    y = [[] for _ in range(len(idx_batch))]
    for i in range(len(idx_batch)):
        for j in range(batch_size):
            x[i].append(data[idx_batch[i][j]][2])
            y[i].append(data[idx_batch[i][j]][3])
    return np.array(x), np.array(y)



class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, data, batch_size=20):
        # Add words to the dictionary
        tokens = 0
        for line in data:
            words = line.split() + ['<eos>']
            tokens += len(words)
            for word in words: 
                self.dictionary.add_word(word)  
        
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        for line in data:
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]
        return ids.view(batch_size, -1)

















