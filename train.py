import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('workingDirectoryPath/AE')
import numpy as np
from data_generator import DataGenerator
from callbacks import Monitor
from se_em_model import *
from metrics import Metrics
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

# prep
batch_size=32
timesteps=70
features=2916
shuffle=True
valSamp=151
IOB=False
mem=[0.77,0.78,0.79]
lr=0.0005
decay_value=0.1
if IOB:
    n_classes=3
else:
    n_classes=2
partition={}
labels={}

# embeddings path
data_path='pathToDirectory/train/'
# labels path
y_train=np.load('pathToDirectory/training_labels.npy')

if !IOB:
    for i,x in enumerate(y_train):
      for j,c in enumerate(x):
        if c==2:
          y_train[i][j]=1

params = {'batch_size': batch_size,
          'timesteps': timesteps,
          'features': features,
          'n_classes': n_classes,
          'shuffle': shuffle,
          'path':data_path,
          'norm':1.0}

partition['train']=['id-{}'.format(x) for x in range(0,y_train.shape[0]-valSamp) ]
partition['validation']=['id-{}'.format(x) for x in range(y_train.shape[0]-valSamp,y_train.shape[0])]
train_labels={ 'id-{}'.format(id):x for id,x in enumerate(y_train[0:y_train.shape[0]-valSamp]) }
val_labels={ 'id-{}'.format(id):y_train[id] for id in range(y_train.shape[0]-valSamp,y_train.shape[0]) }
train_generator = DataGenerator(partition['train'], train_labels, **params)

val_x = np.empty((valSamp, timesteps,features))
val_y = np.empty((valSamp,timesteps), dtype=int)
for i, ID in enumerate(val_labels.keys()):
    v=np.load(params['path'] + ID + '.npy')
    if v.shape[0] !=timesteps:
      v=np.concatenate((v,[np.zeros(features ,dtype='float32')]*(timesteps-v.shape[0])), axis=0)
    val_x[i,] = v
    val_y[i,] = val_labels[ID]
validation_data=(val_x,val_y)

# train
img_path="pathToDirectory/model.png"
checkpoint_path='pathToDirectory'

optimizer=tf.keras.optimizers.RMSprop(
    learning_rate=lr,
    epsilon=1e-08,
    name="RMSprop"
)

callbacks = [
  Monitor(validation_data,save_best_only=True,save=True,patience=15,lr=lr,decay_value=decay_value,
          early_stopping=True,path=checkpoint_path,mem=mem)
]

use_tpu = False
if use_tpu:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    with strategy.scope():
        model = nn_model(img_path,features,optimizer,timesteps,n_classes)
else:
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model = nn_model(img_path,features,optimizer,timesteps,n_classes)

history = model.fit(x=train_generator, epochs=150,callbacks=callbacks,
                    use_multiprocessing=True,workers=16)

"""
################################################################
Epoch 26/150
90/90 [==============================] - ETA: 0s - loss: 0.0013
156 Aspects observed
134 B-A correctly predicted
131 Aspects correctly predicted
B-A accuracy: 85.89743598682%
Accuracy: 83.97435876953896%
F1-score: 82.12244897959184%
Recall-score: 79.96820349761526%
Precision-score: 84.39597315436241%
Memore State:  [0.78, 0.79, 0.7920792079207921]
################################################################
"""
