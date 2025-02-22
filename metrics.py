import tensorflow as tf
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as cp
from tensorflow.keras import backend as K

idx2tag={1:'B-A', 2:'I-A', 0:'O'}
tag2idx={'B-A':1, 'I-A':2, 'O':0}

class Metrics():
  def __init__(self,model,X_test,y_true,IOB=False):
    self.model=model
    self.X_test=X_test
    self.y_pred=K.argmax(model.predict(self.X_test),axis=2).numpy()
    #self.y_true=K.argmax(y_true,axis=2).numpy()
    self.y_true=y_true    
    if not IOB:
      for i,x in enumerate(self.y_pred):
        for j,_ in enumerate(x):   
          if (self.y_pred[i][j-1] in [1,2]) and self.y_pred[i][j]==1:
            self.y_pred[i][j]=2
      for i,x in enumerate(self.y_true):
        for j,_ in enumerate(x):   
          if (self.y_true[i][j-1] in [1,2]) and self.y_true[i][j]==1:
            self.y_true[i][j]=2
            
  def reportAE(self,mode='default'):
    y_true=[ [idx2tag[xi]  for xi in x ] for x in self.y_true]
    y_pred=[ [idx2tag[xi]  for xi in x ] for x in self.y_pred]
    return {'F1-Score':f1_score(y_true, y_pred,mode=mode),
            'Recall':recall_score(y_true, y_pred,mode=mode),
            'Precision':precision_score(y_true, y_pred,mode=mode),
            'Classification Report':classification_report(y_true, y_pred,mode=mode)}

  def accuracy(self):
    val_targ = self.y_true
    val_predict = self.y_pred
    correct_aspect_pred,ground_truth,correct_b_a_pred=0,0,0
    # Accuracy
    for idx,x in enumerate(val_targ):
      b_idx=np.where(x==1)[0]
      i_idx=np.where(x==2)[0]
      ground_truth+=len(b_idx)
      b_pred=tf.where(K.equal(val_predict[idx],1))
      i_pred=tf.where(K.equal(val_predict[idx],2))
      con_pred=tf.where(K.equal(val_predict[idx],3))
      for i in b_idx:   
        if i not in b_pred:
          continue
        correct_aspect_pred+=1 
        correct_b_a_pred+=1   
        j=i+1
        while((j in i_idx)):
          if (j not in i_pred) and (j not in con_pred):
            correct_aspect_pred-=1
            break
          j+=1            
    print('\n{} Aspects observed'.format(ground_truth))
    print('{} B-A correctly predicted'.format(correct_b_a_pred))
    print('{} Aspects correctly predicted'.format(correct_aspect_pred))
    print('B-A accuracy: {}%'.format((correct_b_a_pred/ground_truth)*100.0))
    print('Accuracy: {}%'.format((correct_aspect_pred/ground_truth)*100.0))
  
  def print_cm(self):
    labels=['O', 'B-A', 'I-A']
    y= self.y_true.ravel()
    p= self.y_pred.ravel()
    cm=confusion_matrix(y,p)
    print(cp(y,p))
    print("\n")
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5]) 
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        sum = 0
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            sum =  sum + int(cell)
            print(cell, end=" ")
        print(sum)
