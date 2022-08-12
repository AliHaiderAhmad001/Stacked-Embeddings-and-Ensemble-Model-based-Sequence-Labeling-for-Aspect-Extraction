import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout,Bidirectional,Conv1D,concatenate,add
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from tensorflow_addons.layers import CRF as CF
def nn_model(img_path,n_features,optimizer,timesteps,n_classes):
    img_path=img_path
    x = tf.keras.Input(shape=(timesteps,n_features),
                        name="StackedEmbeddings")  

    conv1_1 = Conv1D(128, 3,padding='same',strides=1,
                      activation='gelu',name='conv1_1')(x)
    conv1_2 = Conv1D(128, 3,padding='same',strides=1,
                      activation='gelu',name='conv1_2')(conv1_1)
    conv1_3 = Conv1D(256, 3,padding='same',strides=1,
                      activation='gelu',name='conv1_3')(conv1_2)
    conv1_4 = Conv1D(256, 3,padding='same',strides=1,
                      activation='gelu',name='conv1_4')(conv1_3)
    conv1_5 = Conv1D(256, 3,padding='same',strides=1,
                      activation='gelu',name='conv1_5')(conv1_4)
    conv1_6 = Conv1D(512, 3,padding='same',strides=1,
                      activation='gelu',name='conv1_6')(conv1_5)
    conv1_7 = Conv1D(512, 5,padding='same',strides=1,
                      activation='gelu',name='conv1_7')(conv1_6)
    conv1_8 = Conv1D(512, 5,padding='same',strides=1,
                      activation='gelu',name='conv1_8')(conv1_7)
    conv1_9 = Conv1D(1024, 5,padding='same',strides=1,
                      activation='gelu',name='conv1_9')(conv1_8)
    conv1_10 = Conv1D(1024, 5,padding='same',strides=1,
                      activation='gelu',name='conv1_10')(conv1_9)
    conv1_11 = Conv1D(1024, 5,padding='same',strides=1,
                      activation='gelu',name='conv1_11')(conv1_10)

    bi_LSTM2_1 = Bidirectional(LSTM(units=512, return_sequences=True,name='Bi_LSTM2_1',activation='tanh',
                                    recurrent_activation = 'sigmoid',recurrent_dropout=0.0,unroll=False,
                                    use_bias=True))(x)
    bi_LSTM2_2 = Bidirectional(LSTM(units=256, return_sequences=True,name='Bi_LSTM2_2',activation='tanh',
                                    recurrent_activation = 'sigmoid',recurrent_dropout=0.0,unroll=False,
                                    use_bias=True))(bi_LSTM2_1)
    bi_LSTM2_3 = Bidirectional(LSTM(units=256, return_sequences=True,name='Bi_LSTM2_3',activation='tanh',
                                    recurrent_activation = 'sigmoid',recurrent_dropout=0.0,unroll=False,
                                    use_bias=True))(bi_LSTM2_2)

    concatenated = concatenate([conv1_11,bi_LSTM2_3], axis=-1)
    drop1=Dropout(0.4)(concatenated)
    fc1 = TimeDistributed(Dense(512, activation="gelu"))(drop1)
    drop2=Dropout(0.3)(fc1)
    fc2 = TimeDistributed(Dense(256, activation="gelu"))(drop2)
    _, potentials, _, _ = CF(n_classes)(fc2)

    model = Model(x, potentials)

    model.compile(optimizer=optimizer,
            loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.7),
            metrics=None) 
    
    tf.keras.utils.plot_model(model, img_path, show_shapes=True)
    return model






