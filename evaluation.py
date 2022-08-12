#!pip install seqeval
#!pip install tensorflow_addons
import numpy as np
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/AE')
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout,Bidirectional,Conv1D,concatenate,add
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from tensorflow_addons.layers import CRF as CF
from tensorflow.keras.models import load_model
from metrics import Metrics

def eval(model_path,data_path,label_path,IOB=False):
  model = load_model(model_path)
  stacked_test_emb=np.load(data_path)
  y_test=np.load(label_path)
  if not IOB:
    for i,x in enumerate(y_test):
      for j,c in enumerate(x):
        if c==2:
          y_test[i][j]=1
  metrics=Metrics(model,stacked_test_emb,y_test)
  print('F1-Score: ',metrics.reportAE()['F1-Score'])
  print(metrics.reportAE()['Classification Report'])
  print('--------------------------------------------------------------')
  print(metrics.print_cm())
  print('--------------------------------------------------------------')
  print(metrics.accuracy())
  print('\n')

# Re14-Eval
print('*******************************************************************')
print('Re14-Evaluation:')
print('*******************************************************************')
model_path='/content/drive/MyDrive/Colab Notebooks/AE/models/Re14Model/final_model'
data_path='/content/drive/MyDrive/Colab Notebooks/AE/embeddings/Re14DataEmb/test/test_data.npy'
label_path='/content/drive/MyDrive/Colab Notebooks/AE/embeddings/Re14Labels/test_labels.npy'
eval(model_path,data_path,label_path)

# Re16-Eval
print('*******************************************************************')
print('Re16-Evaluation:')
print('*******************************************************************')
model_path='/content/drive/MyDrive/Colab Notebooks/AE/models/best/final_model'
data_path='/content/drive/MyDrive/Colab Notebooks/AE/embeddings/Re16DataEmb/test/test_data.npy'
label_path='/content/drive/MyDrive/Colab Notebooks/AE/embeddings/Re16Labels/test_labels.npy'
eval(model_path,data_path,label_path)

# La14-Eval
print('*******************************************************************')
print('laptop14-Evaluation:')
print('*******************************************************************')
model_path='/content/drive/MyDrive/Colab Notebooks/AE/models/LaptopModel/final_model'
data_path='/content/drive/MyDrive/Colab Notebooks/AE/embeddings/LaptopDataEmb/test/test_data.npy'
label_path='/content/drive/MyDrive/Colab Notebooks/AE/embeddings/LaptopDataLabels/test_labels.npy'
eval(model_path,data_path,label_path)
