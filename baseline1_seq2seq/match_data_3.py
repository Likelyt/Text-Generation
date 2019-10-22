#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:29:16 2018

@author: liyt
"""

def Match_data_pair(text, l, sentence_0, sentence_1, sentence_2, sentence_3):
    
    # Convert all text sentence large than 4 to 4
    for i in range(len(l)):
        if l[i] > 4:
            l[i] = 4
    
    data_0 = []
    data_1 = []
    i = 0
    j = 0
    k = 0
    s = 0
    for i in range(len(l)):
        if l[i] == 1:
            data_0.append([sentence_0[i]])
        if l[i] == 2:
            data_0.append([sentence_0[i]])
            data_1.append([sentence_0[i], sentence_1[j]])
            j += 1
        if l[i] == 3:
            data_0.append([sentence_0[i]])
            data_1.append([sentence_0[i], sentence_1[j]])
            data_1.append([sentence_1[j], sentence_2[k]])
            j += 1
            k += 1
        if l[i] == 4:
            data_0.append([sentence_0[i]])
            data_1.append([sentence_0[i], sentence_1[j]])
            data_1.append([sentence_1[j], sentence_2[k]])
            data_1.append([sentence_2[k], sentence_3[s]])
            j += 1
            k += 1
            s += 1

    return data_0, data_1





