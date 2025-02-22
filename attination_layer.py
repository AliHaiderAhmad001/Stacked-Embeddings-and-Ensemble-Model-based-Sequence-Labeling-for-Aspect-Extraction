from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
class attention(Model,Layer):
    def __init__(self, return_sequences=True,name=None,**kwargs):
        super(attention, self).__init__(**kwargs)
        self.return_sequences = return_sequences
  
    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
        super(attention,self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

    def get_config(self):
        config = super(attention, self).get_config().copy()
        config.update({"return_sequences": self.return_sequences})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
