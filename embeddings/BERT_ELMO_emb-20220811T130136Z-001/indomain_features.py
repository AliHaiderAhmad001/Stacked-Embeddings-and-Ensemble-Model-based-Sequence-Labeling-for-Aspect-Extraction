from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import numpy as np
from flair.embeddings import WordEmbeddings

class transformerEmb():
  def __init__(self,model='yangheng/deberta-v3-large-absa-v1.1',subtoken_pooling='mean',
               layers='-1,-2,-3,-4',layer_mean=True):
    self.model = TransformerWordEmbeddings(model,subtoken_pooling=subtoken_pooling,layers=layers,layer_mean=layer_mean)
  def getEmb(self,sent):
    li=[]
    sentence = Sentence(sent)
    self.model.embed(sentence)
    for token in sentence:
      li.append(token.embedding.cpu().numpy())
    return np.array(li)

class wordEmb():
  def __init__(self,model='glove'):
    self.model = WordEmbeddings('glove')
  def getEmb(self,sent):
    li=[]
    sentence = Sentence(sent)
    self.model.embed(sentence)
    for token in sentence:
      li.append(token.embedding.cpu().numpy())
    return np.array(li)
