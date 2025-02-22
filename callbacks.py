import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from seqeval.metrics import precision_score, recall_score, f1_score
import numpy as np

tag2idx={1:'B-A', 2:'I-A', 0:'O'}

class Monitor(Callback):
  def __init__(self,validation_data=None,save=True,save_best_only=True,patience=0,
  early_stopping=False,path='',lr=0.0005,decay_value=0.1,mem=[0.70,0.75,0.79]):
      super(Monitor, self).__init__()
      self.validation_data=validation_data
      self.save_best_only=save_best_only
      self.patience=patience
      self.early_stopping=early_stopping
      self.path=path
      self.save=save
      self.x=patience
      self.lrdecay=True
      self.decay_value=decay_value
      self.lr=lr
      self.memory=mem
    
  def on_epoch_end(self, epoch, logs={}): 
    val_targ = self.validation_data[1]
    val_predict=K.argmax(self.model.predict(self.validation_data[0]),axis=2).numpy()    
    for i,x in enumerate(val_predict):
      for j,_ in enumerate(x):   
        if (val_predict[i][j-1] in [1,2]) and val_predict[i][j]==1:
          val_predict[i][j]=2
    for i,x in enumerate(val_targ):
      for j,_ in enumerate(x):   
        if (val_targ[i][j-1] in [1,2]) and val_targ[i][j]==1:
          val_targ[i][j]=2
    
    correct_aspect_pred,ground_truth,correct_b_a_pred=0,0,0
    # Accuracy&F1-score
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
    val_targ=[ [tag2idx[xi]  for xi in x ] for x in val_targ]
    val_predict=[ [tag2idx[xi]  for xi in x ] for x in val_predict]
    f1=f1_score(val_targ, val_predict)
    rs=recall_score(val_targ, val_predict)
    ps=precision_score(val_targ, val_predict)
    print('F1-score: {}%'.format(f1*100.0))
    print('Recall-score: {}%'.format(rs*100.0))
    print('Precision-score: {}%'.format(ps*100.0))
    
    if f1>self.memory[0] and (self.lrdecay):
      old_lr = self.model.optimizer.lr.read_value()
      self.memory[0]=f1
      self.memory=sorted(self.memory)
      new_lr = old_lr * self.decay_value
      self.model.optimizer.lr.assign(new_lr)
      self.lrdecay=False
      print('Learning-Rate: {}'.format(new_lr))

    if f1<self.memory[0] and (not self.lrdecay):
      self.model.optimizer.lr.assign(self.lr)
      self.lrdecay=True
      print('Learning-Rate: {}'.format(self.lr))
    print('Memore State: ',self.memory)
      
    # early stopping
    if self.early_stopping:
      try:
        f1_old=np.load(self.path+'/f1.npy')
        if f1<f1_old:
          self.x-=1
        else:
          self.x=self.patience
        if self.x==0:
          self.model.stop_training = True
      except:
        pass
    
    # model save
    if self.save:
      if self.save_best_only:
        try:
          f1_old=np.load(self.path+'/f1.npy') 
          if f1>f1_old:
            self.model.save(self.path+'/final_model')
            np.save(self.path+'/f1.npy',np.array(f1))  
            print('The model has been saved at epoch #{}'.format(epoch+1)) 
        except:
            np.save(self.path+'/f1.npy',np.array(f1))
      else:
        self.model.save(self.path+'/model-{}'.format(epoch+1))
        print('The model has been saved at epoch #{}'.format(epoch+1))
    print('################################################################\n')
