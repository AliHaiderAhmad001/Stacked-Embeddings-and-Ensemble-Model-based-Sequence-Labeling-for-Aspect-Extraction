# -*- coding: utf-8 -*-
"""prep_embedding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/174jF0btH3qoTvBW_4BihlpG569DNtMJo
"""

import numpy as np
import json
def gen_np_embedding(fn, word_idx_fn, out_fn, dim=300):
    with open(word_idx_fn) as f:
        word_idx = json.load(f)
    embedding = np.zeros((len(word_idx)+2, dim))
    with open(fn) as f:
        for l in f:
            rec = l.rstrip().split(' ')
            if len(rec) == 2:  # skip the first line.
                continue
            if rec[0] in word_idx:
                embedding[word_idx[rec[0]]] = np.array([float(r) for r in rec[1:]])
    with open(out_fn+".oov.txt", "w") as fw:
        for w in word_idx:
            if embedding[word_idx[w]].sum() == 0.:
                fw.write(w+"\n")
    np.save(out_fn+".npy", embedding.astype('float32'))
restaurant_emb_raw='/home/ali/Desktop/AspectBasedSentimentAnalysis_SE14/data/restaurant_emb.vec'
word_idx_file='/home/ali/Desktop/AspectBasedSentimentAnalysis_SE14/data/word_idx.json'
restaurant_emb='/home/ali/Desktop/AspectBasedSentimentAnalysis_SE14/data/restaurant_emb'
# gen_np_embedding(restaurant_emb_raw,word_idx_file,restaurant_emb,dim=100)
# emb=np.load('/home/ali/Desktop/AspectBasedSentimentAnalysis_SE14/data/restaurant_emb.npy')
# emb[word_idx['pizza']]