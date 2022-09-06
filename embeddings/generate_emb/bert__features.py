#!pip install transformers
# Extracting features from a BERT model
from re import M
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
class BERTModelFeatures():
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.n_gpu =  torch.cuda.device_count()
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).eval().to(self.device)
    self.vocab=self.tokenizer.get_vocab()
    self.features=None

  # Run the text through BERT, and collect all of the hidden states produced from all 12 layers. 
  def runBert(self,indexed_tokens):
    tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
    #attention_masks = torch.tensor([attention_masks]).to(self.device)
    with torch.no_grad():
        outputs = self.model(tokens_tensor)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
    return hidden_states,token_embeddings
  
  def prapearingInput(self,sentence):
    marked_text = "[CLS] " + sentence.lower() + " [SEP]"
    tokenized_text = self.tokenizer.tokenize(marked_text)
    #padding_len=(max_len-len(tokenized_text))
    #if len(tokenized_text)<max_len:
      #tokenized_text=tokenized_text+['']*padding_len # padding to max len
    indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
    #attention_masks = [float(1)]*(max_len-padding_len)+[float(0)]*padding_len
    return tokenized_text,indexed_tokens,marked_text

  # creating the word vectors by summing together the last four layers.
  def tokenVecSum(self,sentence):
    tokenized_text,indexed_tokens,marked_text=self.prapearingInput(sentence)
    hidden_states,token_embeddings=self.runBert(indexed_tokens)
    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec.cpu().numpy())
    words=marked_text.split()
    oov=list(set(words).difference(self.vocab))
    idx=[x for x in range(0,len(tokenized_text))]
    for w in oov:
        w_idx=np.where(np.array(words)==w)[0]
        sub_words_len=len(self.tokenizer.tokenize(w))
        inc=0
        for i in w_idx:
          i+=inc
          for j in range(1,sub_words_len):
            words.insert(i+j,'')
          inc+=sub_words_len-1

    for w in oov:
      w_idx=np.where(np.array(words)==w)[0]
      sub_words_len=len(self.tokenizer.tokenize(w))
      for i in w_idx:
          for j in range(1,sub_words_len):
            token_vecs_sum[i]+=token_vecs_sum[i+j]
            idx.remove(i+j)
    idx.remove(0)
    idx.remove(idx[-1])
    token_vecs_sum=np.array(token_vecs_sum)[idx]
    self.features={'hidden_states':hidden_states,
          'token_embeddings':token_embeddings,
          'tokenized_text':tokenized_text,
          'indexed_tokens':indexed_tokens,
          'token_vecs_sum':token_vecs_sum}
    return self.features

  # creating the word vectors by concatenate together the last four layers.
  def tokenVecCat(self,sentence):
    tokenized_text,indexed_tokens,marked_text=self.prapearingInput(sentence)
    hidden_states,token_embeddings=self.runBert(indexed_tokens)
    token_vecs_cat = []
    for token in token_embeddings:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_vecs_cat.append(cat_vec.cpu().numpy())
        oov=list(set(marked_text.split()).difference(self.vocab))
    idx=[x for x in range(0,len(tokenized_text))]
    for w in oov:
      w_idx=np.where(np.array(marked_text.split())==w)[0]
      sub_words_len=len(self.tokenizer.tokenize(w))
      for i in w_idx:
          for j in range(1,sub_words_len):
            token_vecs_cat[i]+=token_vecs_cat[i+j]
            idx.remove(i+j)
    idx.remove(0)
    idx.remove(idx[-1])
    token_vecs_sum=np.array(token_vecs_sum)[idx]
    
    self.features={'hidden_states':hidden_states,
          'token_embeddings':token_embeddings,
          'tokenized_text':tokenized_text,
          'indexed_tokens':indexed_tokens,
          'token_vecs_cat':token_vecs_cat}
    return self.features

  def generate_bert_features(self,data,type='sum'):
    if type=='sum':
       embeddings =[self.tokenVecSum(" ".join(sentence))['token_vecs_sum'] for sentence in data]
    else:
       embeddings =[self.tokenVecCat(" ".join(sentence))['token_vecs_cat'] for sentence in data]
    return embeddings
  
  # Calculate the cosine similarity between the vectors 
  def vecSimilarity(self,vec1,vec2):
    return (1 - cosine(vec1, vec2))
  
  # Returns a token vector for a specific token
  def token2Vec(self,tokenIdx=None,sum=True):
    tokenized_text=self.features['tokenized_text']
    for i, token_str in enumerate(tokenized_text):
        print (i, token_str)
    tokenIdx=int(input())
    print(self.features['token_vecs_sum'][tokenIdx])
    if sum:
      return self.features['token_vecs_sum'][tokenIdx]
    return self.features['token_vecs_cat'][tokenIdx]
