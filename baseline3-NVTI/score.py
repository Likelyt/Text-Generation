#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:14:30 2018

@author: liyt
"""
import numpy as np
from collections import defaultdict
from Bleu import score_corpus

def cal_entropy(generated):
    etp_score = [0.0,0.0,0.0,0.0]
    div_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for gg in generated:
        g = gg.rstrip('2').split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) +1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) /total
    return etp_score, div_score



def score_calculator(y_hat, y_true):
    """
    Caculating a list of 12 scores, averaged on the predictions.
    :param predictions: a matrix of [bs, len].
    :param ground_truth_sentences: a [bs, max_tgt_len] tensor.
    :return:
    """
    predictions = y_hat.t() # each sentence is row based [4,5,6,7]
    ground_truth_sentences = y_true.t()
    generated = []
    reference = []
    EOS_ID = 2
    for idx in range(len(predictions)):
        pred = list(predictions[idx])
        if EOS_ID in pred:
            pred = pred[:pred.index(EOS_ID)]
        truth = list(ground_truth_sentences[idx])# [1:] # the first SOS should be excluded
        truth = truth[:truth.index(EOS_ID)]

        generated.append(" ".join([str(i) for i in pred]))
        reference.append(" ".join([str(i) for i in truth]))
    
    bleu_scores =[0,0,0,0]
    for i in range(1,5):
        bleu_scores[i-1] = score_corpus(generated, reference, ngrams=i)
        #bleu_scores[i-1] = score_sentence(generated, reference, ngrams=i, smooth=0)
    etp_scores, div_scores = cal_entropy(generated)
    # scores = bleu_scores + etp_scores + div_scores
    return bleu_scores, etp_scores, div_scores





