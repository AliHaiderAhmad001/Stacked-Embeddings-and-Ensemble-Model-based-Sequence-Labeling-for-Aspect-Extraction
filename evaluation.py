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

"""
*******************************************************************
Re14-Evaluation:
*******************************************************************
F1-Score:  0.8937728937728938
              precision    recall  f1-score   support

           A       0.89      0.90      0.89      1088

   micro avg       0.89      0.90      0.89      1088
   macro avg       0.89      0.90      0.89      1088
weighted avg       0.89      0.90      0.89      1088

--------------------------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     54366
           1       0.91      0.92      0.91      1088
           2       0.91      0.88      0.89       546

    accuracy                           1.00     56000
   macro avg       0.94      0.93      0.94     56000
weighted avg       1.00      1.00      1.00     56000



              O   B-A   I-A 
        O 54275    67    24 54366
      B-A    64   999    25 1088
      I-A    36    30   480 546
None
--------------------------------------------------------------

1088 Aspects observed
999 B-A correctly predicted
983 Aspects correctly predicted
B-A accuracy: 91.81985294117648%
Accuracy: 90.34926470588235%
None


*******************************************************************
Re16-Evaluation:
*******************************************************************
F1-Score:  0.7982905982905983
              precision    recall  f1-score   support

           A       0.77      0.82      0.80       567

   micro avg       0.77      0.82      0.80       567
   macro avg       0.77      0.82      0.80       567
weighted avg       0.77      0.82      0.80       567

--------------------------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     45972
           1       0.80      0.85      0.83       567
           2       0.83      0.66      0.73       291

    accuracy                           0.99     46830
   macro avg       0.88      0.84      0.85     46830
weighted avg       0.99      0.99      0.99     46830



              O   B-A   I-A 
        O 45840   106    26 45972
      B-A    71   483    13 567
      I-A    86    14   191 291
None
--------------------------------------------------------------

567 Aspects observed
483 B-A correctly predicted
470 Aspects correctly predicted
B-A accuracy: 85.18518518518519%
Accuracy: 82.89241622574956%
None


*******************************************************************
laptop14-Evaluation:
*******************************************************************
F1-Score:  0.8301282051282051
              precision    recall  f1-score   support

           A       0.84      0.82      0.83       629

   micro avg       0.84      0.82      0.83       629
   macro avg       0.84      0.82      0.83       629
weighted avg       0.84      0.82      0.83       629

--------------------------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     62778
           1       0.87      0.85      0.86       629
           2       0.96      0.78      0.86       433

    accuracy                           1.00     63840
   macro avg       0.94      0.88      0.91     63840
weighted avg       1.00      1.00      1.00     63840



              O   B-A   I-A 
        O 62722    47     9 62778
      B-A    88   536     5 629
      I-A    58    36   339 433
None
--------------------------------------------------------------

629 Aspects observed
536 B-A correctly predicted
519 Aspects correctly predicted
B-A accuracy: 85.21462639109699%
Accuracy: 82.51192368839428%
None



"""
