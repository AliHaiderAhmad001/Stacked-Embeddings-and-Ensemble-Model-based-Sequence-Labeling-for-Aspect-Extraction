import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32,timesteps=70, features=1892,
                 n_classes=3, shuffle=True,path='',norm=1.0):
        # Leave the n_classes= 3 in all cases (e.g IO or IOB)
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.timesteps=timesteps
        self.features = features
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path1 = path
        self.norm=norm
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((self.batch_size, self.timesteps,self.features))
        y = np.zeros((self.batch_size,self.timesteps), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            v=np.load(self.path1 + ID + '.npy')
            if v.shape[0] !=self.timesteps:
              v=np.concatenate((v,[np.zeros(self.features ,dtype='float32')]*(self.timesteps-v.shape[0])), axis=0)
            X[i,] = v
            # Store class
            y[i,] = self.labels[ID]
        y=np.reshape(y, (self.batch_size,self.timesteps,1))
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
